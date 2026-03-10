"""
Structure-Aware Document Parser — Pre-chunking analysis pipeline.

Runs before TextChunker to extract structural information from documents:
  • StructureAnalyzer  — orchestrates all detectors, returns DocumentStructure
  • TablePreserver     — extracts tables as atomic units (PDF/DOCX/TXT only)
  • HeadingDetector    — builds heading hierarchy (PDF/DOCX/TXT only)
  • BoundaryDetector   — identifies logical section boundaries

Note: Markdown/HTML heading detection is already handled by
Chunking._chunk_markdown() and Chunking._chunk_html().
This module targets formats that lack built-in structure (PDF text, DOCX text, TXT).
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import bisect
from src.utils.Logger import get_logger

def _compute_line_offsets(text: str) -> List[int]:
    offsets = [0]
    for i, ch in enumerate(text):
        if ch == '\n':
            offsets.append(i + 1)
    return offsets

logger = get_logger(__name__)

# ── Pre-compiled patterns ───────────────────────────────────────────────

# Heading patterns for plain-text / PDF-extracted text
# Matches ALL-CAPS lines (likely section titles), numbered sections, etc.
ALLCAPS_HEADING = re.compile(r'^[A-Z][A-Z\s\-:]{4,}$', re.MULTILINE)
NUMBERED_HEADING = re.compile(
    r'^(\d+(?:\.\d+)*)\s+([A-Z][\w\s\-:,]{3,})$', re.MULTILINE
)
# "Chapter N", "Section N", "Part N" headings
CHAPTER_HEADING = re.compile(
    r'^(?:chapter|section|part)\s+\d+[\s:\-]*(.*)',
    re.IGNORECASE | re.MULTILINE,
)
# Page break markers (form-feed or common PDF page-break artifacts)
PAGE_BREAK = re.compile(r'\f|\x0c|(?:^-{3,}\s*page\s+\d+\s*-{3,}$)', re.IGNORECASE | re.MULTILINE)

# Table patterns (pipe-separated, tab-separated, multi-space-separated)
TABLE_PIPE_ROW = re.compile(r'^.+\|.+\|.+$', re.MULTILINE)
TABLE_TAB_ROW = re.compile(r'^.+\t.+\t.+$', re.MULTILINE)
MULTI_SPACE = re.compile(r'  +')

# Markdown heading (for pre-detection — actual chunking handled elsewhere)
MD_HEADING = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)


# ── Dataclasses ─────────────────────────────────────────────────────────

@dataclass
class Heading:
    """A detected heading in the document."""
    text: str
    level: int              # 1 = top-level, 2, 3, …
    line_index: int         # line number in the document (0-indexed)
    char_offset: int        # character offset from start of document
    confidence: float = 1.0 # confidence score (1.0 = highly certain)


@dataclass
class TableBlock:
    """An atomic table extracted from the document."""
    text: str
    start_char: int
    end_char: int
    row_count: int


@dataclass
class SectionBoundary:
    """A logical break point in the document."""
    char_offset: int
    boundary_type: str      # "page_break", "heading", "topic_shift", "blank_region"
    label: Optional[str] = None


@dataclass
class DocumentStructure:
    """Complete structural analysis of a document, used to guide the chunker."""
    headings: List[Heading] = field(default_factory=list)
    tables: List[TableBlock] = field(default_factory=list)
    boundaries: List[SectionBoundary] = field(default_factory=list)
    heading_tree: List[dict] = field(default_factory=list)  # nested H1→H2→H3 tree
    has_structure: bool = False   # True if any structural features were detected
    section_spans: List[Tuple[int, int, Optional[str]]] = field(default_factory=list)
    table_spans: List[Tuple[int, int]] = field(default_factory=list)
    table_starts: List[int] = field(default_factory=list)


# ── TablePreserver ──────────────────────────────────────────────────────

class TablePreserver:
    """Extract tables as atomic units from raw text.

    Detects pipe-separated, tab-separated, and multi-space aligned tables.
    A table must have ≥ 2 rows to qualify.
    """

    MIN_TABLE_ROWS = 2

    def extract_tables(self, text: str, line_offsets: Optional[List[int]] = None) -> List[TableBlock]:
        """Identify and extract all table blocks in the text.

        Returns:
            List of TableBlock objects sorted by position.
        """
        if not text:
            return []
            
        if line_offsets is None:
            line_offsets = _compute_line_offsets(text)

        lines = text.split('\n')
        tables: List[TableBlock] = []

        in_table = False
        table_start_idx = 0
        table_lines: List[str] = []

        for i, line in enumerate(lines):
            is_table_row = self._is_table_row(line)

            if is_table_row:
                if not in_table:
                    in_table = True
                    table_start_idx = i
                    table_lines = [line]
                else:
                    table_lines.append(line)
            else:
                if in_table and len(table_lines) >= self.MIN_TABLE_ROWS:
                    tables.append(self._build_table_block(table_start_idx, table_lines, line_offsets))
                in_table = False
                table_lines = []

        # Handle table at end of text
        if in_table and len(table_lines) >= self.MIN_TABLE_ROWS:
            tables.append(self._build_table_block(table_start_idx, table_lines, line_offsets))

        if tables:
            logger.debug(f"TablePreserver: found {len(tables)} tables")

        return tables

    @staticmethod
    def _is_table_row(line: str) -> bool:
        """Check if a line looks like a table row."""
        if not line.strip():
            return False

        if TABLE_PIPE_ROW.match(line):
            return True
        if TABLE_TAB_ROW.match(line):
            return True

        # safer multi-space heuristic
        tokens = line.strip().split()
        if len(tokens) >= 3:
            gaps = MULTI_SPACE.findall(line)
            if len(gaps) >= 2:
                avg_token_len = sum(len(t) for t in tokens) / len(tokens)
                if avg_token_len <= 12:  # filters paragraphs
                    return True

        return False

    @staticmethod
    def _build_table_block(start_idx: int, table_lines: List[str], line_offsets: List[int]) -> TableBlock:
        """Build a TableBlock from detected table lines."""
        table_text = '\n'.join(table_lines)
        start_char = line_offsets[start_idx]
        end_char = start_char + len(table_text)
        return TableBlock(
            text=table_text,
            start_char=start_char,
            end_char=end_char,
            row_count=len(table_lines),
        )


# ── HeadingDetector ─────────────────────────────────────────────────────

class HeadingDetector:
    """Build a heading hierarchy tree from document text.

    Targets PDF/DOCX/TXT where headings are not marked up.
    Uses heuristics: ALL-CAPS lines, numbered sections (1.2.3 Title),
    and chapter/section markers.
    """

    def detect_headings(self, text: str, file_ext: str = "", line_offsets: Optional[List[int]] = None, tables: Optional[List[TableBlock]] = None) -> List[Heading]:
        """Detect headings in text.

        Args:
            text: Document text.
            file_ext: File extension (e.g. ".pdf", ".md"). Markdown/HTML
                      headings are detected but NOT used for chunking
                      (Chunking.py handles those natively).

        Returns:
            Sorted list of Heading objects.
        """
        if not text:
            return []
            
        if line_offsets is None:
            line_offsets = _compute_line_offsets(text)

        headings: List[Heading] = []
        table_spans = sorted([(t.start_char, t.end_char) for t in (tables or [])], key=lambda x: x[0])
        table_starts = [ts[0] for ts in table_spans]
        
        def is_in_table(char_offset: int) -> bool:
            return StructureAnalyzer.is_inside_any_table(char_offset, table_spans, table_starts)

        # 1. Markdown headings (for structure map only)
        if file_ext.lower() in ('.md', '.markdown'):
            for m in MD_HEADING.finditer(text):
                if is_in_table(m.start()): continue
                level = len(m.group(1))
                headings.append(Heading(
                    text=m.group(2).strip(),
                    level=level,
                    line_index=bisect.bisect_right(line_offsets, m.start()) - 1,
                    char_offset=m.start(),
                    confidence=1.0,
                ))
            return sorted(headings, key=lambda h: h.char_offset)

        # 2. Chapter / Section markers  →  level 1
        for m in CHAPTER_HEADING.finditer(text):
            if is_in_table(m.start()): continue
            headings.append(Heading(
                text=m.group(0).strip(),
                level=1,
                line_index=bisect.bisect_right(line_offsets, m.start()) - 1,
                char_offset=m.start(),
                confidence=0.95,
            ))

        # 3. Numbered headings (1.2.3 Title)  →  level = depth of numbering
        for m in NUMBERED_HEADING.finditer(text):
            if is_in_table(m.start()): continue
            number_part = m.group(1)
            level = number_part.count('.') + 1
            headings.append(Heading(
                text=m.group(0).strip(),
                level=min(level, 4),  # cap at level 4
                line_index=bisect.bisect_right(line_offsets, m.start()) - 1,
                char_offset=m.start(),
                confidence=0.9,
            ))

        # 4. ALL-CAPS lines (likely section titles)  →  level 2
        for m in ALLCAPS_HEADING.finditer(text):
            if is_in_table(m.start()): continue
            candidate = m.group(0).strip()
            
            # NEW filters
            if len(candidate.split()) > 12:
                continue
            if candidate.endswith('.'):
                continue
            if candidate.count(':') > 2:
                continue
                
            # Filter out noise: skip lines that are too short or too long
            if 5 <= len(candidate) <= 80:
                headings.append(Heading(
                    text=candidate,
                    level=2,
                    line_index=bisect.bisect_right(line_offsets, m.start()) - 1,
                    char_offset=m.start(),
                    confidence=0.6,
                ))

        # Deduplicate by char_offset (multiple patterns may match same line)
        # Keep the highest-confidence candidate at each offset.
        best_by_offset = {}
        for h in headings:
            prev = best_by_offset.get(h.char_offset)
            if prev is None or h.confidence > prev.confidence:
                best_by_offset[h.char_offset] = h
        unique_headings = sorted(best_by_offset.values(), key=lambda h: h.char_offset)

        if unique_headings:
            logger.debug(f"HeadingDetector: found {len(unique_headings)} headings")

        return unique_headings

    def build_heading_tree(self, headings: List[Heading]) -> List[dict]:
        """Build a nested tree from flat heading list.

        Returns:
            List of dicts: { "text": str, "level": int, "children": [...] }
        """
        if not headings:
            return []

        root: List[dict] = []
        stack: List[dict] = []  # (node, level) tracking

        for h in headings:
            node = {"text": h.text, "level": h.level, "char_offset": h.char_offset, "children": []}

            # Pop everything in stack with level >= current
            while stack and stack[-1]["level"] >= h.level:
                stack.pop()

            if stack:
                stack[-1]["children"].append(node)
            else:
                root.append(node)

            stack.append(node)

        return root


# ── BoundaryDetector ────────────────────────────────────────────────────

class BoundaryDetector:
    """Identify logical section boundaries in text.

    Detects:
      • Page breaks (form-feed characters, explicit markers)
      • Headings (provided by HeadingDetector)
      • Blank-line regions (≥ 3 consecutive blank lines = topic shift)
    """

    BLANK_LINE_THRESHOLD = 3  # consecutive blanks to qualify as boundary

    def detect_boundaries(
        self,
        text: str,
        headings: Optional[List[Heading]] = None,
    ) -> List[SectionBoundary]:
        """Detect all logical boundaries.

        Returns:
            Sorted list of SectionBoundary objects.
        """
        if not text:
            return []

        boundaries: List[SectionBoundary] = []

        # 1. Page breaks
        for m in PAGE_BREAK.finditer(text):
            boundaries.append(SectionBoundary(
                char_offset=m.start(),
                boundary_type="page_break",
            ))

        # 2. Heading-based boundaries
        if headings:
            for h in headings:
                boundaries.append(SectionBoundary(
                    char_offset=h.char_offset,
                    boundary_type="heading",
                    label=h.text,
                ))

        # 3. Large blank-line regions (topic shifts)
        lines = text.split('\n')
        consecutive_blanks = 0
        char_pos = 0
        for line in lines:
            if line.strip() == '':
                consecutive_blanks += 1
            else:
                if consecutive_blanks >= self.BLANK_LINE_THRESHOLD:
                    boundaries.append(SectionBoundary(
                        char_offset=char_pos,
                        boundary_type="blank_region",
                    ))
                consecutive_blanks = 0
            char_pos += len(line) + 1

        # Sort and deduplicate (bucketed dedupe via 50-char window)
        boundaries.sort(key=lambda b: b.char_offset)
        
        DEDUP_WINDOW = 50
        seen = set()
        deduped: List[SectionBoundary] = []

        for b in boundaries:
            bucket = b.char_offset // DEDUP_WINDOW
            dedupe_key = (bucket, b.boundary_type, b.label)
            if dedupe_key not in seen:
                seen.add(dedupe_key)
                deduped.append(b)

        return deduped


# ── StructureAnalyzer (main entry point) ────────────────────────────────

class StructureAnalyzer:
    """Orchestrates all structure detectors and returns a DocumentStructure.

    Usage:
        analyzer = StructureAnalyzer()
        structure = analyzer.detect_document_structure(text, ".pdf")
    """

    def __init__(self):
        self.table_preserver = TablePreserver()
        self.heading_detector = HeadingDetector()
        self.boundary_detector = BoundaryDetector()

    def _compute_section_spans(self, text: str, headings: List[Heading]) -> List[Tuple[int, int, Optional[str]]]:
        if not headings:
            text_len = len(text)
            return [(0, text_len, None)] if text_len > 0 else []

        # Headings are already sorted by char_offset from detection, but sort again to be absolutely sure
        sorted_headings = sorted(headings, key=lambda h: h.char_offset)
        spans: List[Tuple[int, int, Optional[str]]] = []
        text_len = len(text)
        
        # Capture text before first heading
        if sorted_headings[0].char_offset > 200:
            spans.append((0, sorted_headings[0].char_offset, None))
            
        for i, h in enumerate(sorted_headings):
            start = h.char_offset
            end = sorted_headings[i+1].char_offset if i + 1 < len(sorted_headings) else text_len
            
            start = max(0, min(start, text_len))
            end = max(0, min(end, text_len))
            
            if start < end:
                spans.append((start, end, h.text))
                
        return spans

    def _adjust_sections_for_tables(self, sections: List[Tuple[int, int, Optional[str]]], table_spans: List[Tuple[int, int]], text_length: int) -> List[Tuple[int, int, Optional[str]]]:
        if not sections or not table_spans:
            return sections

        table_spans = sorted(table_spans, key=lambda x: x[0])
        table_starts = [ts[0] for ts in table_spans]

        adjusted_sections: List[Tuple[int, int, Optional[str]]] = []
        for start, end, label in sections:
            new_start, new_end = start, end

            # Check table potentially containing section start
            idx_start = bisect.bisect_right(table_starts, new_start)
            if idx_start > 0:
                t_start, t_end = table_spans[idx_start - 1]
                if t_start < new_start < t_end:
                    new_start = t_start

            # Check table potentially containing section end
            idx_end = bisect.bisect_right(table_starts, new_end)
            if idx_end > 0:
                t_start, t_end = table_spans[idx_end - 1]
                if t_start < new_end < t_end:
                    new_end = t_end

            # Clamp to document bounds
            new_start = max(0, min(new_start, text_length))
            new_end = max(0, min(new_end, text_length))

            if new_start < new_end:
                adjusted_sections.append((new_start, new_end, label))

        # Normalize overlaps after boundary shifts.
        final_sections: List[Tuple[int, int, Optional[str]]] = []
        for start, end, label in sorted(adjusted_sections, key=lambda s: s[0]):
            if not final_sections:
                final_sections.append((start, end, label))
                continue

            prev_start, prev_end, prev_label = final_sections[-1]

            # Merge contiguous/overlapping ranges for same label.
            if start <= prev_end and label == prev_label:
                final_sections[-1] = (prev_start, max(prev_end, end), prev_label)
                continue

            # Prevent overlap for different labels.
            if start < prev_end:
                start = prev_end

            if start < end:
                final_sections.append((start, end, label))

        return final_sections

    @staticmethod
    def is_inside_any_table(char_offset: int, table_spans: List[Tuple[int, int]], table_starts: List[int]) -> bool:
        """O(log n) check if offset is inside any table. table_spans must be sorted by start."""
        if not table_spans or not table_starts:
            return False
        # Use bisect to find the first table that starts AFTER our char_offset
        idx = bisect.bisect_right(table_starts, char_offset)
        # If idx == 0, all tables start after char_offset
        if idx == 0:
            return False
        # The table that could contain char_offset is at idx - 1
        t_start, t_end = table_spans[idx - 1]
        return t_start <= char_offset < t_end

    def detect_document_structure(self, text: str, file_ext: str = "") -> DocumentStructure:
        """Run the full structure analysis pipeline.

        Args:
            text: Raw document text.
            file_ext: File extension (e.g. ".pdf", ".txt", ".md").

        Returns:
            DocumentStructure with all detected structural elements.
        """
        if not text or not text.strip():
            return DocumentStructure(has_structure=False)
            
        if len(text) < 500 and text.count('\n') < 8 and len(text.split()) < 120:
            # Very short docs rarely benefit from heavy structure analysis.
            return DocumentStructure(has_structure=False)

        line_offsets = _compute_line_offsets(text)

        # 1. Detect tables
        tables = self.table_preserver.extract_tables(text, line_offsets=line_offsets)
        table_spans = sorted([(t.start_char, t.end_char) for t in tables], key=lambda x: x[0])
        table_starts = [ts[0] for ts in table_spans]

        # 2. Detect headings
        headings = self.heading_detector.detect_headings(text, file_ext, line_offsets=line_offsets, tables=tables)
        heading_tree = self.heading_detector.build_heading_tree(headings)

        # 3. Detect boundaries (uses headings for heading-based breaks)
        boundaries = self.boundary_detector.detect_boundaries(text, headings)

        # 4. Compute optimal section spans
        section_spans = self._compute_section_spans(text, headings)
        section_spans = self._adjust_sections_for_tables(section_spans, table_spans, len(text))

        has_structure = bool(
            tables or 
            headings or 
            boundaries or 
            len(section_spans) > 1
        )

        structure = DocumentStructure(
            headings=headings,
            tables=tables,
            boundaries=boundaries,
            heading_tree=heading_tree,
            has_structure=has_structure,
            section_spans=section_spans,
            table_spans=table_spans,
            table_starts=table_starts,
        )

        if has_structure:
            logger.info(
                f"DocumentStructure: "
                f"{len(headings)} headings, "
                f"{len(tables)} tables, "
                f"{len(boundaries)} boundaries, "
                f"{len(section_spans)} sections"
            )
        else:
            logger.debug("No document structure detected")

        return structure
