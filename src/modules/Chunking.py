"""
Text Chunking Module for RAG Systems - Enterprise Scale

Provides multiple chunking strategies with streaming, batch processing,
and incremental indexing for handling millions of documents.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Generator, Iterator
import json
import os
from pathlib import Path
from datetime import datetime
import logging
import psycopg2
import psycopg2.extras
import threading
import re
import ast
import csv
import io
import hashlib
from collections import OrderedDict
import sys

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from src.utils.Logger import get_logger
from src.modules.Loader import DocumentMetadata, Document
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

logger = get_logger(__name__)

# Pre-compiled regex patterns for performance (avoids repeated compilation)
HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+')
PARAGRAPH_PATTERN = re.compile(r'\n\s*\n')
WHITESPACE_PATTERN = re.compile(r'\s+')
# Table detection patterns
TABLE_ROW_PATTERN = re.compile(r'^.+[|\t].+[|\t].+$', re.MULTILINE)  # Rows with | or tabs
TABLE_MULTISPACE_PATTERN = re.compile(r'^.+\s{2,}.+\s{2,}.+$', re.MULTILINE)  # Rows with multiple spaces


class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    HEADER_BASED = "header_based"        # Markdown
    DOM_AWARE = "dom_aware"              # HTML
    CODE_STRUCTURE = "code_structure"    # Code files
    ROW_GROUP = "row_group"              # CSV/Excel
    KEY_PATH = "key_path"                # JSON/YAML
    CELL_AWARE = "cell_aware"            # Jupyter notebooks


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk"""
    document_id: str
    document_name: str
    chunk_index: int
    char_count: int
    token_count: int
    start_char: Optional[int]
    end_char: Optional[int]
    chunk_id: str = ""
    strategy: str = "fixed_size"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    # Extension-specific metadata
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    header_hierarchy: Optional[List[str]] = None
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    row_range: Optional[Tuple[int, int]] = None
    key_path: Optional[str] = None
    cell_type: Optional[str] = None
    char_offsets_approximate: bool = False


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    id: str
    text: str
    metadata: ChunkMetadata


class TextChunker:
    """
    Scalable text chunking for millions of documents.
    
    Optimizations:
    - Streaming chunk generation (generator-based)
    - SQLite backend for chunk persistence
    - Batch processing with incremental indexing
    - Token counting cache with LRU
    - Resume capability for interrupted chunking
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        default_strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE,
        encoding_name: str = "cl100k_base",
        min_chunk_size: int = 50,
        chunks_dir: Optional[str] = None,
        use_db: bool = True,
        db_path: Optional[str] = None
    ):
        """
        Initialize the TextChunker
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Token overlap between chunks
            default_strategy: Default chunking strategy (overridden by file extension if recognized)
            encoding_name: Tokenizer encoding name
            min_chunk_size: Minimum chunk size in tokens
            chunks_dir: Directory to save chunk metadata
            use_db: Use SQLite for chunk storage
            db_path: Path to SQLite database
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.default_strategy = default_strategy if isinstance(default_strategy, ChunkingStrategy) else ChunkingStrategy(default_strategy)
        self.encoding_name = encoding_name
        self.min_chunk_size = min_chunk_size
        self.chunks_dir = chunks_dir or os.path.join(os.path.dirname(__file__), "../../data/chunks")
        
        # Database configuration
        self.use_db = use_db
        # For PostgreSQL, db_path is not used (network connection)
        self.db_path = db_path  # Kept for backward compatibility, not used
        
        # Initialize tokenizer if available
        self.tokenizer = None
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                logger.warning(f"Could not initialize tiktoken: {e}")
        
        # Statistics
        self._documents_processed = 0
        self._chunks_created = 0
        self._db_conn: Optional[psycopg2.extensions.connection] = None
        self._db_lock = threading.Lock()
        
        # Token count cache (hash -> count) with OrderedDict for efficient FIFO
        # Increased from 10K to 50K for better hit rates on large corpora
        self._token_cache: OrderedDict[str, int] = OrderedDict()
        self._max_cache_size = 50000  # Configurable cache size
        self._cache_hits = 0
        self._cache_misses = 0
        
        Path(self.chunks_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize database if needed
        if self.use_db:
            self._init_database()
        
        logger.info(f"TextChunker initialized: default_strategy={self.default_strategy.value}, chunk_size={chunk_size}, use_db={use_db}")
    
    def _init_database(self) -> None:
        """Initialize PostgreSQL database for chunk storage"""
        try:
            conn = psycopg2.connect(
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                database=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                connect_timeout=10
            )
            conn.autocommit = False
            cursor = conn.cursor()
            
            # Create chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT,
                    document_name TEXT,
                    chunk_index INTEGER,
                    text TEXT,
                    char_count INTEGER,
                    token_count INTEGER,
                    start_char INTEGER,
                    end_char INTEGER,
                    strategy TEXT,
                    metadata_json TEXT,
                    created_at TEXT,
                    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(id)
                )
            """)
            
            # Create index for fast lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunk_doc_id 
                ON chunks(document_id)
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f"PostgreSQL chunks database initialized: {POSTGRES_DB}@{POSTGRES_HOST}:{POSTGRES_PORT}")
        except psycopg2.Error as e:
            logger.error(f"PostgreSQL error initializing chunks database: {e}")
            self.use_db = False
        except Exception as e:
            logger.error(f"Error initializing chunks database: {e}")
            self.use_db = False
    
    def _get_db_connection(self) -> psycopg2.extensions.connection:
        """Get thread-safe database connection with optimized settings"""
        if self._db_conn is None or self._db_conn.closed:
            self._db_conn = psycopg2.connect(
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                database=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                connect_timeout=10
                # No statement_timeout for writes - only for reads to avoid batch insert slowdowns
            )
            self._db_conn.autocommit = False
        return self._db_conn
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens using hash-based cache (prevents memory duplication)"""
        if not text:
            return 0
        
        # Early estimation for very short text (optimization: skip hashing)
        text_len = len(text)
        if text_len < 20:  # Very short text, use fast heuristic
            return max(1, text_len // 4)
        
        # Use hash of text as cache key to save memory
        text_hash = hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()
        
        # Check cache
        if text_hash in self._token_cache:
            self._cache_hits += 1
            return self._token_cache[text_hash]
        
        self._cache_misses += 1
        
        # Compute token count
        if self.tokenizer:
            try:
                count = len(self.tokenizer.encode(text))
            except Exception:
                count = max(1, text_len // 4)
        else:
            # Ultra-fast fallback: ~4 chars per token
            count = max(1, text_len // 4)
        
        # Cache with configurable size limit (FIFO with OrderedDict)
        if len(self._token_cache) >= self._max_cache_size:
            # Remove oldest item efficiently
            self._token_cache.popitem(last=False)
        
        self._token_cache[text_hash] = count
        return count
    
    def get_token_count(self, text: str) -> int:
        """Get token count for text
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        return self._count_tokens(text)
    
    def _chunk_fixed_size(self, text: str, doc_id: str) -> List[Tuple[str, int, int]]:
        """Chunk text into fixed-size chunks by tokens"""
        if not text:
            return []
        
        chunks = []
        
        # If text has no line breaks, split by words instead
        if '\n' not in text:
            words = text.split()
            current_chunk = []
            current_tokens = 0
            start_idx = 0
            char_offset = 0  # Track character position incrementally
            
            for i, word in enumerate(words):
                word_tokens = self._count_tokens(word + ' ')
                
                if current_tokens + word_tokens > self.chunk_size and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    start_char = char_offset
                    end_char = start_char + len(chunk_text)
                    chunks.append((chunk_text, start_char, end_char))
                    
                    # Calculate offset for next chunk
                    chunk_size_with_space = len(chunk_text) + 1  # +1 for space
                    char_offset += chunk_size_with_space
                    
                    # Reset for next chunk
                    current_chunk = []
                    current_tokens = 0
                    start_idx = i
                
                current_chunk.append(word)
                current_tokens += word_tokens
            
            # Add final chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                start_char = char_offset
                end_char = start_char + len(chunk_text)
                chunks.append((chunk_text, start_char, end_char))
        
        else:
            # Original line-based splitting for documents with newlines
            lines = text.split('\n')
            current_chunk = []
            current_tokens = 0
            start_char = 0
            char_pos = 0
            
            for line in lines:
                line_tokens = self._count_tokens(line + '\n')
                
                # If adding this line exceeds chunk size, save current chunk
                if current_tokens + line_tokens > self.chunk_size and current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append((chunk_text, start_char, char_pos))
                    
                    # Prepare overlap with cached token count
                    overlap_lines = current_chunk[-max(1, len(current_chunk) // 2):]
                    current_chunk = overlap_lines
                    current_tokens = sum(self._count_tokens(l + '\n') for l in overlap_lines)
                    start_char = char_pos - sum(len(l) + 1 for l in overlap_lines)
                
                current_chunk.append(line)
                current_tokens += line_tokens
                char_pos += len(line) + 1
            
            # Add final chunk
            if current_chunk:
                chunk_text = '\n'.join(current_chunk)
                chunks.append((chunk_text, start_char, char_pos))
        
        return chunks
    
    def _chunk_sliding_window(self, text: str, doc_id: str) -> List[Tuple[str, int, int]]:
        """Chunk text using sliding window"""
        if not text:
            return []
        
        chunks = []
        words = text.split()
        chunk_words = []
        start_idx = 0
        
        for i, word in enumerate(words):
            chunk_words.append(word)
            current_tokens = self._count_tokens(' '.join(chunk_words))
            
            # If chunk is full, save it
            if current_tokens >= self.chunk_size:
                chunk_text = ' '.join(chunk_words)
                # Calculate character positions
                start_char = sum(len(w) + 1 for w in words[:start_idx])
                end_char = start_char + len(chunk_text)
                chunks.append((chunk_text, start_char, end_char))
                
                # Slide window by overlap
                overlap_words = int(len(chunk_words) * self.chunk_overlap / self.chunk_size)
                start_idx = max(start_idx + 1, i - overlap_words + 1)
                chunk_words = words[start_idx:i+1]
        
        # Add final chunk
        if chunk_words:
            chunk_text = ' '.join(chunk_words)
            start_char = sum(len(w) + 1 for w in words[:start_idx])
            end_char = start_char + len(chunk_text)
            chunks.append((chunk_text, start_char, end_char))
        
        return chunks
    
    def _chunk_paragraph(self, text: str, doc_id: str) -> List[Tuple[str, int, int]]:
        """Chunk text by paragraphs"""
        if not text:
            return []
        
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_tokens = 0
        char_pos = 0
        start_char = 0
        
        for para in paragraphs:
            para_tokens = self._count_tokens(para)
            
            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append((chunk_text, start_char, char_pos))
                current_chunk = []
                current_tokens = 0
                start_char = char_pos + 2
            
            current_chunk.append(para)
            current_tokens += para_tokens
            char_pos += len(para) + 2
        
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append((chunk_text, start_char, char_pos))
        
        return chunks
    
    def _detect_table_blocks(self, text: str) -> List[Tuple[int, int, str]]:
        """Detect table blocks in text and return their positions
        
        Returns:
            List of (start_pos, end_pos, table_text) tuples
        """
        tables = []
        lines = text.split('\n')
        
        in_table = False
        table_start_idx = 0
        table_lines = []
        
        for i, line in enumerate(lines):
            # Check if line looks like a table row
            has_pipe = '|' in line and line.count('|') >= 2
            has_tabs = '\t' in line and line.count('\t') >= 2
            # Multi-space: check for 2+ consecutive spaces separating at least 3 columns
            has_multi_space = len(re.findall(r'  +', line)) >= 2 and len(line.split()) >= 3
            
            is_table_row = has_pipe or has_tabs or has_multi_space
            
            if is_table_row:
                if not in_table:
                    # Start new table
                    in_table = True
                    table_start_idx = i
                    table_lines = [line]
                else:
                    # Continue current table
                    table_lines.append(line)
            else:
                if in_table and len(table_lines) >= 2:  # Minimum 2 rows to be a table
                    # End current table
                    table_text = '\n'.join(table_lines)
                    # Calculate character positions
                    start_pos = sum(len(lines[j]) + 1 for j in range(table_start_idx))
                    end_pos = start_pos + len(table_text)
                    tables.append((start_pos, end_pos, table_text))
                
                in_table = False
                table_lines = []
        
        # Handle table at end of text
        if in_table and len(table_lines) >= 2:
            table_text = '\n'.join(table_lines)
            start_pos = sum(len(lines[j]) + 1 for j in range(table_start_idx))
            end_pos = start_pos + len(table_text)
            tables.append((start_pos, end_pos, table_text))
        
        return tables
    
    def _chunk_semantic_text(self, text: str, doc_id: str, doc_name: str = "") -> List[Dict]:
        """Semantic text chunking for PDF/DOCX/TXT - 600 tokens, 15% overlap
        
        Now with table-aware chunking: Tables are detected and kept as atomic units.
        """
        if not text:
            return []
        
        chunks = []
        target_tokens = 600
        overlap_tokens = int(target_tokens * 0.15)  # 15% overlap = 90 tokens
        
        # Detect tables and replace them with placeholders
        table_blocks = self._detect_table_blocks(text)
        table_map = {}  # placeholder -> table_text
        modified_text = text
        
        # Replace tables with placeholders (from end to start to preserve positions)
        for idx, (start, end, table_text) in enumerate(reversed(table_blocks)):
            placeholder = f"<<<TABLE_{len(table_blocks)-idx-1}>>>"
            table_map[placeholder] = table_text
            modified_text = modified_text[:start] + placeholder + modified_text[end:]
        
        # Split into paragraphs (no header detection for semantic chunking - too error-prone)
        paragraphs = modified_text.split('\n\n')
        
        current_chunk = []
        current_tokens_list = []  # Store (sentence, tokens) for efficient overlap
        current_tokens = 0
        current_heading = None
        char_pos = 0
        start_char = 0
        
        for para in paragraphs:
            # Check if paragraph contains a table placeholder
            if '<<<TABLE_' in para:
                for placeholder, table_text in table_map.items():
                    if placeholder in para:
                        # Extract table
                        table_tokens = self._count_tokens(table_text)
                        
                        # If current chunk + table is too large, save current chunk first
                        if current_chunk and current_tokens + table_tokens > target_tokens:
                            chunk_text = ' '.join(current_chunk)
                            chunks.append({
                                'text': chunk_text,
                                'start_char': start_char,
                                'end_char': start_char + len(chunk_text),
                                'metadata': {
                                    'section_title': current_heading,
                                    'char_offsets_approximate': True
                                }
                            })
                            current_chunk = []
                            current_tokens_list = []
                            current_tokens = 0
                            start_char = start_char + len(chunk_text)
                        
                        # Add table with markers
                        table_with_markers = f"\n[TABLE]\n{table_text}\n[/TABLE]\n"
                        current_chunk.append(table_with_markers)
                        current_tokens_list.append((table_with_markers, table_tokens))
                        current_tokens += table_tokens
                        
                        # If table is large, save as its own chunk
                        if table_tokens > target_tokens * 0.7:
                            chunk_text = ' '.join(current_chunk)
                            chunks.append({
                                'text': chunk_text,
                                'start_char': start_char,
                                'end_char': start_char + len(chunk_text),
                                'metadata': {
                                    'section_title': current_heading,
                                    'contains_table': True,
                                    'char_offsets_approximate': True
                                }
                            })
                            current_chunk = []
                            current_tokens_list = []
                            current_tokens = 0
                            start_char = start_char + len(chunk_text)
                        break
                continue  # Skip regular paragraph processing
            
            # Regular paragraph processing (no table)
            # Split long paragraphs by sentences
            sentences = SENTENCE_PATTERN.split(para)  # Use pre-compiled pattern
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                # Use pre-compiled pattern for performance
                sentence_tokens = self._count_tokens(sentence)
                
                if current_tokens + sentence_tokens > target_tokens and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'start_char': start_char,
                        'end_char': start_char + len(chunk_text),
                        'metadata': {
                            'section_title': current_heading,
                            'char_offsets_approximate': True
                        }
                    })
                    
                    # Keep overlap for context using precomputed tokens
                    overlap_chunk = []
                    overlap_tokens_count = 0
                    for s, s_tokens in reversed(current_tokens_list):
                        if overlap_tokens_count + s_tokens > overlap_tokens:
                            break
                        overlap_chunk.insert(0, s)
                        overlap_tokens_count += s_tokens
                    
                    current_chunk = overlap_chunk
                    current_tokens_list = [(s, self._count_tokens(s)) for s in overlap_chunk]
                    current_tokens = sum(t for _, t in current_tokens_list)
                    start_char = start_char + len(chunk_text) - len(' '.join(overlap_chunk))
                
                current_chunk.append(sentence)
                current_tokens_list.append((sentence, sentence_tokens))
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'start_char': start_char,
                'end_char': start_char + len(chunk_text),
                'metadata': {
                    'section_title': current_heading,
                    'char_offsets_approximate': True
                }
            })
        
        return chunks
    
    def _chunk_markdown(self, text: str, doc_id: str) -> List[Dict]:
        """Header-based chunking for Markdown files"""
        if not text:
            return []
        
        chunks = []
        # Split by headers (# ## ###)
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = text.split('\n')
        
        current_section = []
        current_headers = []
        current_level = 0
        start_char = 0
        char_pos = 0
        
        for line in lines:
            match = re.match(header_pattern, line)
            
            if match:
                # Save previous section
                if current_section:
                    section_text = '\n'.join(current_section)
                    section_tokens = self._count_tokens(section_text)
                    
                    # If section is too large, fall back to text chunker
                    if section_tokens > self.chunk_size * 2:
                        sub_chunks = self._chunk_fixed_size(section_text, doc_id)
                        for sub_text, sub_start, sub_end in sub_chunks:
                            chunks.append({
                                'text': sub_text,
                                'start_char': start_char + sub_start,
                                'end_char': start_char + sub_end,
                                'metadata': {
                                    'header_hierarchy': current_headers.copy(),
                                    'char_offsets_approximate': True
                                }
                            })
                    else:
                        chunks.append({
                            'text': section_text,
                            'start_char': start_char,
                            'end_char': char_pos,
                            'metadata': {
                                'header_hierarchy': current_headers.copy(),
                                'char_offsets_approximate': True
                            }
                        })
                
                # Start new section
                level = len(match.group(1))
                title = match.group(2).strip()
                
                # Update header hierarchy
                if level <= current_level:
                    current_headers = current_headers[:level-1]
                current_headers.append(title)
                current_level = level
                
                current_section = [line]
                start_char = char_pos
            else:
                current_section.append(line)
            
            char_pos += len(line) + 1
        
        # Add final section
        if current_section:
            section_text = '\n'.join(current_section)
            chunks.append({
                'text': section_text,
                'start_char': start_char,
                'end_char': char_pos,
                'metadata': {
                    'header_hierarchy': current_headers.copy(),
                    'char_offsets_approximate': True
                }
            })
        
        return chunks
    
    def _chunk_html(self, text: str, doc_id: str) -> List[Dict]:
        """DOM-aware chunking for HTML files with header hierarchy tracking"""
        if not BS4_AVAILABLE:
            logger.warning("BeautifulSoup not available, falling back to text chunking")
            return [{'text': t[0], 'start_char': t[1], 'end_char': t[2], 'metadata': {}} 
                    for t in self._chunk_fixed_size(text, doc_id)]
        
        chunks = []
        try:
            soup = BeautifulSoup(text, 'html.parser')
            
            # Remove boilerplate
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()
            
            # Extract content by headings and paragraphs
            content_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'article', 'section'])
            
            current_chunk = []
            current_tokens = 0
            header_hierarchy = []
            
            for elem in content_elements:
                elem_text = elem.get_text(strip=True)
                if not elem_text:
                    continue
                
                # Track header hierarchy
                if elem.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    level = int(elem.name[1])
                    header_hierarchy = header_hierarchy[:level-1]
                    header_hierarchy.append(elem_text)
                
                elem_tokens = self._count_tokens(elem_text)
                
                if current_tokens + elem_tokens > self.chunk_size and current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'start_char': None,
                        'end_char': None,
                        'metadata': {
                            'header_hierarchy': header_hierarchy.copy(),
                            'char_offsets_approximate': True
                        }
                    })
                    current_chunk = []
                    current_tokens = 0
                
                current_chunk.append(elem_text)
                current_tokens += elem_tokens
            
            # Add final chunk
            if current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'start_char': None,
                    'end_char': None,
                    'metadata': {
                        'header_hierarchy': header_hierarchy.copy(),
                        'char_offsets_approximate': True
                    }
                })
        
        except Exception as e:
            logger.warning(f"HTML parsing failed: {e}, falling back to text chunking")
            return [{'text': t[0], 'start_char': t[1], 'end_char': t[2], 'metadata': {}} 
                    for t in self._chunk_fixed_size(text, doc_id)]
        
        return chunks
    
    def _chunk_code(self, text: str, doc_id: str, file_ext: str) -> List[Dict]:
        """Structure-aware chunking for code files"""
        chunks = []
        seen_ranges = set()  # Track line ranges to avoid duplicates
        
        try:
            if file_ext == '.py':
                # Parse Python code
                tree = ast.parse(text)
                lines = text.split('\n')
                
                # Use iter_child_nodes for top-level only (avoids nested duplicates)
                for node in ast.iter_child_nodes(tree):
                    chunk_text = None
                    metadata = {}
                    start_line = end_line = None
                    
                    if isinstance(node, ast.FunctionDef):
                        # Extract function with signature + docstring + body
                        start_line = node.lineno - 1
                        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
                        
                        # Check if already seen
                        range_key = (start_line, end_line)
                        if range_key in seen_ranges:
                            continue
                        seen_ranges.add(range_key)
                        
                        chunk_text = '\n'.join(lines[start_line:end_line])
                        metadata = {
                            'function_name': node.name,
                            'type': 'function'
                        }
                    
                    elif isinstance(node, ast.ClassDef):
                        # Extract class definition
                        start_line = node.lineno - 1
                        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 20
                        
                        # Check if already seen
                        range_key = (start_line, end_line)
                        if range_key in seen_ranges:
                            continue
                        seen_ranges.add(range_key)
                        
                        chunk_text = '\n'.join(lines[start_line:end_line])
                        metadata = {
                            'class_name': node.name,
                            'type': 'class'
                        }
                    
                    if chunk_text:
                        chunks.append({
                            'text': chunk_text,
                            'start_char': None,  # Line-based, not char-based
                            'end_char': None,
                            'metadata': {**metadata, 'start_line': start_line, 'end_line': end_line}
                        })
            
            else:
                # For other languages, use pattern-based extraction
                # Match function/class declarations
                patterns = [
                    r'(?:public|private|protected)?\s*(?:static)?\s*(?:async)?\s*\w+\s+\w+\s*\([^)]*\)\s*\{[^}]*\}',  # Java/C#/JS functions
                    r'function\s+\w+\s*\([^)]*\)\s*\{[^}]*\}',  # JavaScript functions
                    r'def\s+\w+\s*\([^)]*\):',  # Python functions (fallback)
                    r'class\s+\w+\s*(?:\([^)]*\))?\s*:',  # Python classes (fallback)
                ]
                
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.DOTALL)
                    for match in matches:
                        chunk_text = match.group(0)
                        chunks.append({
                            'text': chunk_text,
                            'start_char': match.start(),
                            'end_char': match.end(),
                            'metadata': {'type': 'code_block'}
                        })
        
        except Exception as e:
            logger.warning(f"Code parsing failed: {e}, falling back to text chunking")
            return [{'text': t[0], 'start_char': t[1], 'end_char': t[2], 'metadata': {}} 
                    for t in self._chunk_fixed_size(text, doc_id)]
        
        # If no chunks were found, fall back to text chunking
        if not chunks:
            return [{'text': t[0], 'start_char': t[1], 'end_char': t[2], 'metadata': {}} 
                    for t in self._chunk_fixed_size(text, doc_id)]
        
        return chunks
    
    def _chunk_csv(self, text: str, doc_id: str) -> List[Dict]:
        """Row-group chunking for CSV/TSV files - HIGHLY OPTIMIZED for large CSVs
        
        Two modes:
        1. Q&A Mode: Detects query/answer/context columns and chunks per row for semantic retrieval
        2. Bulk Mode: Large row groups (250 rows) with compressed format for efficiency
        
        Optimizations:
        - Auto-detection of Q&A structure
        - Vectorized operations (100x faster than iterrows)
        - Compressed text format for bulk data
        - Skip numeric-only chunks
        """
        if not PANDAS_AVAILABLE:
            logger.warning("Pandas not available, falling back to text chunking")
            return [{'text': t[0], 'start_char': t[1], 'end_char': t[2], 'metadata': {}} 
                    for t in self._chunk_fixed_size(text, doc_id)]
        
        chunks = []
        
        try:
            # Read CSV with optimized settings for large files
            df = None
            
            # Fast parsing with C engine (fastest option)
            try:
                df = pd.read_csv(
                    io.StringIO(text), 
                    engine='c',  # C engine is 10x faster than python
                    low_memory=False,  # Faster for mixed dtypes
                    on_bad_lines='skip'
                )
                if df is not None and not df.empty:
                    logger.debug(f"CSV parsed: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e1:
                # Fallback: Python engine with delimiter sniffing
                try:
                    df = pd.read_csv(
                        io.StringIO(text), 
                        sep=None, 
                        engine='python',
                        on_bad_lines='skip'
                    )
                    logger.debug(f"CSV parsed (fallback): {len(df)} rows")
                except Exception as e2:
                    logger.warning(f"CSV parsing failed: {str(e2)[:100]}")
                    raise
            
            if df is not None and not df.empty:
                # AUTO-DETECT CSV STRUCTURE
                columns_lower = [col.lower() for col in df.columns]
                
                # Detect Document columns
                has_document = any(col in columns_lower for col in ['text', 'content', 'document'])
                
                # Detect Q&A columns
                has_question = any(col in columns_lower for col in ['query', 'question', 'q'])
                has_answer = any(col in columns_lower for col in ['answer', 'response', 'a'])
                has_qa_structure = has_question and has_answer
                
                if has_document:
                    # MODE 1: Document Mode - Chunk the specific text column(s) normally
                    logger.info(f"Document CSV detected - extracting and chunking text column")
                    
                    # Find the primary text column
                    text_col = None
                    for col in df.columns:
                        col_lower = col.lower()
                        if col_lower in ['text', 'content', 'document']:
                            text_col = col
                            break
                    
                    # Identify other columns to treat as metadata (e.g., source_url, index)
                    metadata_cols = [col for col in df.columns if col != text_col]
                    
                    for idx, row in df.iterrows():
                        if pd.notna(row[text_col]) and str(row[text_col]).strip():
                            row_text = str(row[text_col])
                            
                            # Build metadata for this row
                            row_metadata = {'row_index': int(idx), 'csv_mode': 'document'}
                            for m_col in metadata_cols:
                                if pd.notna(row[m_col]):
                                    val = str(row[m_col])
                                    # Truncate massive metadata
                                    row_metadata[m_col] = val if len(val) < 200 else val[:197] + "..."
                            
                            # Use the standard character-size chunker on the cell's text
                            row_chunks = self._chunk_fixed_size(row_text, doc_id)
                            
                            for text_chunk, start_char, end_char in row_chunks:
                                # Attach the row metadata to every sub-chunk
                                chunk_meta = row_metadata.copy()
                                chunks.append({
                                    'text': text_chunk,
                                    'start_char': start_char,
                                    'end_char': end_char,
                                    'metadata': chunk_meta
                                })
                                
                    logger.debug(f"Document CSV created {len(chunks)} chunks from {len(df)} rows")
                
                elif has_qa_structure:
                    # MODE 2: Q&A Mode - One chunk per row for semantic retrieval
                    logger.info(f"Q&A CSV detected - using per-row chunking")
                    
                    # Find relevant columns
                    query_col = next((col for col in df.columns if col.lower() in ['query', 'question', 'q']), None)
                    answer_col = next((col for col in df.columns if col.lower() in ['answer', 'response', 'a']), None)
                    context_col = next((col for col in df.columns if col.lower() in ['context', 'passage', 'text', 'content']), None)
                    index_col = next((col for col in df.columns if col.lower() in ['document_index', 'doc_index', 'doc_id', 'document_id', 'id', 'index', 'idx', 'group', 'topic']), None)
                    
                    # Other metadata columns
                    known_cols = {query_col, answer_col, context_col, index_col}
                    other_cols = [col for col in df.columns if col not in known_cols]
                    
                    for idx, row in df.iterrows():
                        chunk_parts = []
                        
                        if query_col and pd.notna(row[query_col]):
                            chunk_parts.append(f"Question: {row[query_col]}")
                        if answer_col and pd.notna(row[answer_col]):
                            chunk_parts.append(f"Answer: {row[answer_col]}")
                        if context_col and pd.notna(row[context_col]):
                            context_value = str(row[context_col])
                            if len(context_value) > 2000:  # Truncate very long contexts
                                context_value = context_value[:2000] + "..."
                            chunk_parts.append(f"Context: {context_value}")
                        
                        # Add other columns as metadata line
                        if other_cols:
                            meta_parts = [f"{col}: {row[col]}" for col in other_cols if pd.notna(row[col])]
                            if meta_parts:
                                chunk_parts.append("Metadata: " + " | ".join(meta_parts))
                        
                        chunk_text = "\n".join(chunk_parts)
                        
                        if len(chunk_text.strip()) > 20:
                            chunk_metadata = {'row_index': int(idx), 'csv_mode': 'qa'}
                            if index_col and pd.notna(row[index_col]):
                                chunk_metadata['document_index'] = str(row[index_col])
                            
                            chunks.append({
                                'text': chunk_text,
                                'start_char': None,
                                'end_char': None,
                                'metadata': chunk_metadata
                            })
                    
                    logger.debug(f"Q&A CSV created {len(chunks)} chunks")
                    
                else:
                    # MODE 3: Bulk Mode - Large row groups for generic data
                    logger.info(f"Bulk CSV detected - using compressed chunking")
                    rows_per_chunk = 50  # Reduced from 250 to avoid exceeding embedding token limits
                    MAX_CHUNK_CHARS = 25000  # ~6000 tokens, well within API limits
                    
                    # Process in chunks
                    for i in range(0, len(df), rows_per_chunk):
                        chunk_df = df.iloc[i:i+rows_per_chunk]
                        
                        # VECTORIZED OPERATION: Convert all rows to strings at once
                        chunk_text = chunk_df.astype(str).agg(' | '.join, axis=1).str.cat(sep='\n')
                        
                        # Cap chunk size to prevent exceeding embedding API token limits
                        if len(chunk_text) > MAX_CHUNK_CHARS:
                            logger.warning(
                                f"Bulk CSV chunk too large ({len(chunk_text)} chars), "
                                f"truncating to {MAX_CHUNK_CHARS} chars"
                            )
                            chunk_text = chunk_text[:MAX_CHUNK_CHARS]
                        
                        # OPTIMIZATION: Skip numeric-only chunks (fast check)
                        sample = chunk_text[:100].replace('|', '').replace(',', '').replace('\n', '').replace(' ', '').replace('.', '').replace('-', '')
                        is_likely_numeric = sample.replace('nan', '').isdigit() if sample else False
                        
                        if not is_likely_numeric and len(chunk_text.strip()) > 20:
                            chunks.append({
                                'text': chunk_text,
                                'start_char': None,
                                'end_char': None,
                                'metadata': {
                                    'row_range': (i, min(i + rows_per_chunk, len(df))),
                                    'csv_mode': 'bulk',
                                    'char_offsets_approximate': True
                                }
                            })
                    
                    logger.debug(f"Bulk CSV chunking created {len(chunks)} chunks from {len(df)} rows (vectorized)")
                
                return chunks
            else:
                logger.warning("CSV parsing returned empty dataframe")
                raise ValueError("Empty dataframe after parsing")
        
        except Exception as e:
            logger.warning(f"CSV parsing failed: {str(e)[:200]}, falling back to text chunking")
            return [{'text': t[0], 'start_char': t[1], 'end_char': t[2], 'metadata': {}} 
                    for t in self._chunk_fixed_size(text, doc_id)]
        
        return chunks
    
    def _chunk_json(self, text: str, doc_id: str) -> List[Dict]:
        """Key-path chunking for JSON/YAML files"""
        chunks = []
        
        try:
            data = json.loads(text)
            
            def flatten_json(obj, parent_key='', sep='.'):
                """Flatten nested JSON with dot paths"""
                items = []
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        new_key = f"{parent_key}{sep}{k}" if parent_key else k
                        if isinstance(v, (dict, list)):
                            items.extend(flatten_json(v, new_key, sep=sep))
                        else:
                            items.append((new_key, v))
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        new_key = f"{parent_key}[{i}]"
                        if isinstance(item, (dict, list)):
                            items.extend(flatten_json(item, new_key, sep=sep))
                        else:
                            items.append((new_key, item))
                return items
            
            # Chunk per top-level key
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        flattened = flatten_json(value, key)
                        chunk_text = '\n'.join([f"{k}: {v}" for k, v in flattened])
                    else:
                        chunk_text = f"{key}: {value}"
                    
                    chunks.append({
                        'text': chunk_text,
                        'start_char': None,
                        'end_char': None,
                        'metadata': {'key_path': key, 'char_offsets_approximate': True}
                    })
            elif isinstance(data, list):
                # Handle JSON arrays
                for i, item in enumerate(data):
                    flattened = flatten_json(item, f"[{i}]")
                    chunk_text = '\n'.join([f"{k}: {v}" for k, v in flattened])
                    chunks.append({
                        'text': chunk_text,
                        'start_char': None,
                        'end_char': None,
                        'metadata': {'key_path': f"[{i}]", 'char_offsets_approximate': True}
                    })
        
        except Exception as e:
            logger.warning(f"JSON parsing failed: {e}, falling back to text chunking")
            return [{'text': t[0], 'start_char': t[1], 'end_char': t[2], 'metadata': {}} 
                    for t in self._chunk_fixed_size(text, doc_id)]
        
        return chunks
    
    def _chunk_yaml(self, text: str, doc_id: str) -> List[Dict]:
        """Key-path chunking for YAML files"""
        if not YAML_AVAILABLE:
            logger.warning("PyYAML not available, falling back to text chunking")
            return [{'text': t[0], 'start_char': t[1], 'end_char': t[2], 'metadata': {}} 
                    for t in self._chunk_fixed_size(text, doc_id)]
        
        try:
            data = yaml.safe_load(text)
            # Convert to JSON and use JSON chunking
            json_text = json.dumps(data)
            return self._chunk_json(json_text, doc_id)
        except Exception as e:
            logger.warning(f"YAML parsing failed: {e}, falling back to text chunking")
            return [{'text': t[0], 'start_char': t[1], 'end_char': t[2], 'metadata': {}} 
                    for t in self._chunk_fixed_size(text, doc_id)]
    
    def _chunk_notebook(self, text: str, doc_id: str) -> List[Dict]:
        """Cell-aware chunking for Jupyter notebooks"""
        chunks = []
        
        try:
            notebook = json.loads(text)
            cells = notebook.get('cells', [])
            
            for i, cell in enumerate(cells):
                cell_type = cell.get('cell_type', 'unknown')
                source = cell.get('source', [])
                
                # Ignore output cells
                if cell_type == 'code':
                    # Join source lines
                    cell_text = ''.join(source) if isinstance(source, list) else source
                    
                    # Use code chunking for code cells
                    code_chunks = self._chunk_code(cell_text, doc_id, '.py')
                    for chunk in code_chunks:
                        chunk['metadata']['cell_type'] = 'code'
                        chunk['metadata']['cell_index'] = i
                        chunks.append(chunk)
                
                elif cell_type == 'markdown':
                    # Use markdown chunking for markdown cells (better header awareness)
                    cell_text = ''.join(source) if isinstance(source, list) else source
                    md_chunks = self._chunk_markdown(cell_text, doc_id)
                    for chunk in md_chunks:
                        chunk['metadata']['cell_type'] = 'markdown'
                        chunk['metadata']['cell_index'] = i
                        chunks.append(chunk)
        
        except Exception as e:
            logger.warning(f"Notebook parsing failed: {e}, falling back to text chunking")
            return [{'text': t[0], 'start_char': t[1], 'end_char': t[2], 'metadata': {}} 
                    for t in self._chunk_fixed_size(text, doc_id)]
        
        return chunks
    
    def _get_chunking_strategy_by_extension(self, doc_name: str) -> ChunkingStrategy:
        """Determine chunking strategy based on file extension"""
        ext = Path(doc_name).suffix.lower()
        
        # Extension-based routing
        if ext in ['.pdf', '.docx', '.txt']:
            return ChunkingStrategy.SEMANTIC
        elif ext == '.md':
            return ChunkingStrategy.HEADER_BASED
        elif ext in ['.html', '.htm']:
            return ChunkingStrategy.DOM_AWARE
        elif ext in ['.py', '.js', '.java', '.cpp', '.c', '.cs', '.go', '.rs', '.ts', '.jsx', '.tsx']:
            return ChunkingStrategy.CODE_STRUCTURE
        elif ext == '.csv':
            return ChunkingStrategy.ROW_GROUP
        elif ext in ['.xlsx', '.xls']:
            return ChunkingStrategy.ROW_GROUP
        elif ext == '.json':
            return ChunkingStrategy.KEY_PATH
        elif ext in ['.yaml', '.yml']:
            return ChunkingStrategy.KEY_PATH
        elif ext == '.ipynb':
            return ChunkingStrategy.CELL_AWARE
        else:
            return self.default_strategy  # Use configured default
    
    def chunk_text(self, text: str, doc_id: str, doc_name: str) -> List[TextChunk]:
        """
        Chunk a single document text into TextChunk objects with extension-based routing
        
        Args:
            text: Document text to chunk
            doc_id: Document ID
            doc_name: Document name
            
        Returns:
            List of TextChunk objects
        """
        if not text:
            return []
        
        # Determine strategy based on file extension
        file_ext = Path(doc_name).suffix.lower()
        strategy_name = self._get_chunking_strategy_by_extension(doc_name)
        
        # Route to appropriate chunking method
        chunk_dicts = []
        if strategy_name == ChunkingStrategy.SEMANTIC:
            chunk_dicts = self._chunk_semantic_text(text, doc_id, doc_name)
        elif strategy_name == ChunkingStrategy.HEADER_BASED:
            chunk_dicts = self._chunk_markdown(text, doc_id)
        elif strategy_name == ChunkingStrategy.DOM_AWARE:
            chunk_dicts = self._chunk_html(text, doc_id)
        elif strategy_name == ChunkingStrategy.CODE_STRUCTURE:
            chunk_dicts = self._chunk_code(text, doc_id, file_ext)
        elif strategy_name == ChunkingStrategy.ROW_GROUP:
            chunk_dicts = self._chunk_csv(text, doc_id)
        elif strategy_name == ChunkingStrategy.KEY_PATH:
            if file_ext in ['.yaml', '.yml']:
                chunk_dicts = self._chunk_yaml(text, doc_id)
            else:
                chunk_dicts = self._chunk_json(text, doc_id)
        elif strategy_name == ChunkingStrategy.CELL_AWARE:
            chunk_dicts = self._chunk_notebook(text, doc_id)
        else:
            # Default strategies (fixed_size, sliding_window, paragraph)
            if self.default_strategy == ChunkingStrategy.FIXED_SIZE:
                chunk_tuples = self._chunk_fixed_size(text, doc_id)
            elif self.default_strategy == ChunkingStrategy.SLIDING_WINDOW:
                chunk_tuples = self._chunk_sliding_window(text, doc_id)
            elif self.default_strategy == ChunkingStrategy.PARAGRAPH:
                chunk_tuples = self._chunk_paragraph(text, doc_id)
            else:
                chunk_tuples = self._chunk_fixed_size(text, doc_id)
            
            # Convert tuples to dict format
            chunk_dicts = [
                {'text': t[0], 'start_char': t[1], 'end_char': t[2], 'metadata': {}}
                for t in chunk_tuples
            ]
        
        # Convert to TextChunk objects with metadata
        chunks = []
        for idx, chunk_dict in enumerate(chunk_dicts):
            chunk_text = chunk_dict['text']
            
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue
            
            token_count = self._count_tokens(chunk_text)
            chunk_id = f"{doc_id}_chunk_{idx:05d}"
            
            # Extract additional metadata
            extra_metadata = chunk_dict.get('metadata', {})
            
            metadata = ChunkMetadata(
                document_id=doc_id,
                document_name=doc_name,
                chunk_index=idx,
                char_count=len(chunk_text),
                token_count=token_count,
                start_char=chunk_dict['start_char'],
                end_char=chunk_dict['end_char'],
                chunk_id=chunk_id,
                strategy=strategy_name.value,
                section_title=extra_metadata.get('section_title'),
                header_hierarchy=extra_metadata.get('header_hierarchy'),
                function_name=extra_metadata.get('function_name'),
                class_name=extra_metadata.get('class_name'),
                row_range=extra_metadata.get('row_range'),
                key_path=extra_metadata.get('key_path'),
                cell_type=extra_metadata.get('cell_type'),
                char_offsets_approximate=extra_metadata.get('char_offsets_approximate', False)
            )
            
            chunk = TextChunk(id=chunk_id, text=chunk_text, metadata=metadata)
            chunks.append(chunk)
        
        # Update statistics
        self._chunks_created += len(chunks)
        
        return chunks
    
    def chunk_document(self, document: Document, save_chunks: bool = False) -> List[TextChunk]:
        """Convenience method to chunk a Document object
        
        Args:
            document: Document object to chunk
            save_chunks: Whether to save chunks to disk
            
        Returns:
            List of TextChunk objects
        """
        chunks = self.chunk_text(document.content, document.id, document.name)
        
        if chunks and save_chunks:
            self._save_chunks(document.id, chunks)
        
        self._documents_processed += 1
        return chunks
    
    def chunk_document_stream(self, document_id: str, content: str, doc_name: str = "") -> Generator[TextChunk, None, None]:
        """Stream chunks from a single document (memory efficient for large files)
        
        Args:
            document_id: Document ID
            content: Document content text
            doc_name: Document name
            
        Yields:
            TextChunk objects one at a time
            
        Note:
            Uses extension-based routing same as chunk_text() for consistency.
            For truly massive files, this streams chunk generation.
        """
        # Use the same routing logic as chunk_text to ensure consistency
        chunks = self.chunk_text(content, document_id, doc_name)
        
        # Stream the chunks one at a time
        for chunk in chunks:
            yield chunk
    
    def save_chunks_batch(self, chunks: List[TextChunk]) -> bool:
        """Save batch of chunks to database efficiently using executemany()
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            True if successful
        """
        if not self.use_db or not chunks:
            return False
        
        try:
            with self._db_lock:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                # Prepare batch data for executemany (up to 10x faster than loop)
                batch_data = []
                for chunk in chunks:
                    # Serialize extended metadata to JSON
                    metadata_dict = {
                        'section_title': chunk.metadata.section_title,
                        'header_hierarchy': chunk.metadata.header_hierarchy,
                        'function_name': chunk.metadata.function_name,
                        'class_name': chunk.metadata.class_name,
                        'row_range': chunk.metadata.row_range,
                        'key_path': chunk.metadata.key_path,
                        'cell_type': chunk.metadata.cell_type,
                        'char_offsets_approximate': chunk.metadata.char_offsets_approximate
                    }
                    metadata_json = json.dumps(metadata_dict)
                    
                    batch_data.append((
                        chunk.id,
                        chunk.metadata.document_id,
                        chunk.metadata.document_name,
                        chunk.metadata.chunk_index,
                        chunk.text,
                        chunk.metadata.char_count,
                        chunk.metadata.token_count,
                        chunk.metadata.start_char,
                        chunk.metadata.end_char,
                        chunk.metadata.strategy,
                        metadata_json,
                        chunk.metadata.created_at
                    ))
                
                # Batch insert with ON CONFLICT for PostgreSQL (upsert)
                # page_size=1000 means PostgreSQL will batch 1000 rows per network round-trip (10x faster)
                psycopg2.extras.execute_batch(cursor, """
                    INSERT INTO chunks 
                    (chunk_id, document_id, document_name, chunk_index, text, 
                     char_count, token_count, start_char, end_char, strategy, metadata_json, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        document_name = EXCLUDED.document_name,
                        chunk_index = EXCLUDED.chunk_index,
                        text = EXCLUDED.text,
                        char_count = EXCLUDED.char_count,
                        token_count = EXCLUDED.token_count,
                        start_char = EXCLUDED.start_char,
                        end_char = EXCLUDED.end_char,
                        strategy = EXCLUDED.strategy,
                        metadata_json = EXCLUDED.metadata_json,
                        created_at = EXCLUDED.created_at
                """, batch_data, page_size=1000)
                
                conn.commit()
                cursor.close()
                self._chunks_created += len(chunks)
                return True
        except psycopg2.Error as e:
            logger.error(f"PostgreSQL error saving chunks: {e}")
            if conn:
                conn.rollback()
            return False
        except Exception as e:
            logger.error(f"Error saving chunks: {e}")
            if conn:
                conn.rollback()
            return False
    
    def get_chunks_for_document(self, doc_id: str) -> List[Dict]:
        """Retrieve all chunks for a document from database
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of chunk dictionaries
        """
        if not self.use_db:
            return []
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """SELECT chunk_id, document_id, text, token_count, start_char, end_char 
                   FROM chunks WHERE document_id = %s ORDER BY chunk_index""",
                (doc_id,)
            )
            
            chunks = []
            for row in cursor.fetchall():
                chunks.append({
                    "chunk_id": row[0],
                    "document_id": row[1],
                    "text": row[2],
                    "token_count": row[3],
                    "start_char": row[4],
                    "end_char": row[5]
                })
            
            cursor.close()
            return chunks
        except psycopg2.Error as e:
            logger.error(f"PostgreSQL error retrieving chunks: {e}")
            return []
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def get_chunk_count(self) -> int:
        """Get total number of chunks in database
        
        Returns:
            Total chunk count
        """
        if not self.use_db:
            return self._chunks_created
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunks")
            count = cursor.fetchone()[0]
            cursor.close()
            return count
        except psycopg2.Error as e:
            logger.error(f"PostgreSQL error getting chunk count: {e}")
            return self._chunks_created
        except Exception:
            return self._chunks_created
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Get token count cache performance statistics
        
        Returns:
            Dictionary with cache metrics for performance monitoring
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': len(self._token_cache),
            'max_cache_size': self._max_cache_size,
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'total_requests': total_requests,
            'hit_rate_percent': round(hit_rate, 2),
            'memory_saved_calls': self._cache_hits  # Each hit saves expensive tiktoken call
        }
    
    def close(self) -> None:
        """Close database connection"""
        if self._db_conn:
            self._db_conn.close()
            self._db_conn = None
    
    def chunk_documents_batch(
        self,
        documents: List[Document],
        save_chunks: bool = True
    ) -> Dict[str, List[TextChunk]]:
        """
        Chunk multiple documents efficiently in batch with optimizations
        
        Args:
            documents: List of Document objects to chunk
            save_chunks: Whether to save chunk metadata to disk
            
        Returns:
            Dictionary mapping document_id to list of TextChunk objects
        """
        all_chunks = {}
        
        # Pre-filter empty documents
        valid_docs = [doc for doc in documents if doc and doc.content and len(doc.content) > 0]
        
        if not valid_docs:
            return all_chunks
        
        # Process documents with minimal overhead
        for doc in valid_docs:
            chunks = self.chunk_text(doc.content, doc.id, doc.name)
            if chunks:
                all_chunks[doc.id] = chunks
                
                # Optionally save chunk metadata
                if save_chunks:
                    self._save_chunks(doc.id, chunks)
        
        return all_chunks
    
    def _save_chunks(self, doc_id: str, chunks: List[TextChunk]) -> bool:
        """Save chunk metadata to JSON file"""
        try:
            chunks_data = {
                "document_id": doc_id,
                "total_chunks": len(chunks),
                "created_at": datetime.now().isoformat(),
                "chunks": [
                    {
                        "id": chunk.id,
                        "metadata": {
                            "document_id": chunk.metadata.document_id,
                            "document_name": chunk.metadata.document_name,
                            "chunk_index": chunk.metadata.chunk_index,
                            "char_count": chunk.metadata.char_count,
                            "token_count": chunk.metadata.token_count,
                            "start_char": chunk.metadata.start_char,
                            "end_char": chunk.metadata.end_char,
                            "created_at": chunk.metadata.created_at
                        }
                    }
                    for chunk in chunks
                ]
            }
            
            output_file = os.path.join(self.chunks_dir, f"{doc_id}_chunks.json")
            with open(output_file, 'w') as f:
                json.dump(chunks_data, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving chunks for {doc_id}: {e}")
            return False
    
    def load_chunks_metadata(self, doc_id: str) -> Optional[Dict]:
        """Load previously saved chunk metadata from JSON file
        
        Args:
            doc_id: Document ID to load metadata for
            
        Returns:
            Dictionary with chunk metadata or None if not found
        """
        try:
            metadata_file = os.path.join(self.chunks_dir, f"{doc_id}_chunks.json")
            if not os.path.exists(metadata_file):
                return None
            
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading chunks metadata for {doc_id}: {e}")
            return None
    
    def get_statistics(self) -> Dict:
        """Get statistics about chunking"""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "default_strategy": self.default_strategy.value,
            "encoding": self.encoding_name,
            "min_chunk_size": self.min_chunk_size,
            "tokenizer_available": TIKTOKEN_AVAILABLE,
            "documents_processed": self._documents_processed,
            "chunks_created": self._chunks_created,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._token_cache),
            "cache_maxsize": 10000
        }


def create_chunker(config: Dict) -> TextChunker:
    """
    Factory function to create a TextChunker with configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured TextChunker instance
    """
    # Support both 'strategy' and 'default_strategy' for backward compatibility
    strategy_key = 'default_strategy' if 'default_strategy' in config else 'strategy'
    return TextChunker(
        chunk_size=config.get("chunk_size", 512),
        chunk_overlap=config.get("chunk_overlap", 50),
        default_strategy=ChunkingStrategy(config.get(strategy_key, "fixed_size")),
        encoding_name=config.get("encoding_name", "cl100k_base"),
        min_chunk_size=config.get("min_chunk_size", 50),
        chunks_dir=config.get("chunks_dir")
    )
