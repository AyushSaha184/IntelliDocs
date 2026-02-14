import os
import json
import hashlib
import multiprocessing
import psycopg2
import psycopg2.extras
import threading
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set, Generator, Iterator
from dataclasses import dataclass, asdict, field
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
from collections import defaultdict

try:
    from unstructured.partition.auto import partition  # type: ignore
except ImportError:
    partition = None

try:
    import fitz  # PyMuPDF for fast PDF loading (10x faster than unstructured)
    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False

from src.utils.Logger import get_logger
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

logger = get_logger(__name__)

# Configuration - Optimized for millions of files
MAX_WORKERS = min(multiprocessing.cpu_count() or 4, 64)  # Scale up to 64 on modern systems (optimized for fast SSDs)
WARNING_SIZE_MB = 50
BUFFER_SIZE = 65536  # 64KB for file hashing
MAX_FILE_SIZE_MB = 30  # Skip files over 100MB
BATCH_SIZE = 1000  # Process files in batches
CHECKPOINT_INTERVAL = 500  # Save checkpoint every 500 files
MEMORY_LIMIT_MB = 512  # Keep processed docs under 512MB in memory

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    # Documents
    '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.csv',
    # Text files
    '.txt', '.md', '.rst', '.json', '.xml', '.html', '.htm',
    # Code files
    '.py', '.js', '.java', '.cpp', '.c', '.h', '.sh', '.yml', '.yaml'
}

@dataclass
class DocumentMetadata:
    """Lightweight document metadata (without content) for memory efficiency"""
    id: str
    name: str
    path: str
    file_type: str
    pages: int
    file_size: int
    hash: str
    loaded_at: str
    status: str = "loaded"
    content_ref: str = ""  # Reference to content storage (db id, file path, etc.)


@dataclass
class Document:
    """Full document with content - use only when needed"""
    id: str
    name: str
    path: str
    file_type: str
    content: str
    pages: int
    file_size: int
    hash: str
    loaded_at: str
    status: str = "loaded"


class DocumentLoader:
    """
    Scalable document loader for millions of files.
    
    Optimizations:
    - Batch processing with streaming file discovery
    - SQLite backend for metadata and deduplication
    - Memory-efficient design with checkpoint/resume capability
    - Configurable worker scaling
    """
    
    def __init__(self, documents_dir: str = None, use_db: bool = True, db_path: str = None, batch_size: int = BATCH_SIZE):
        """
        Initialize DocumentLoader
        
        Args:
            documents_dir: Path to documents directory. Defaults to data/documents
            use_db: Use SQLite database for metadata (recommended for large scale)
            db_path: Path to SQLite database file. Auto-generated if not specified
            batch_size: Number of files to process before saving checkpoint
        """
        self.documents_dir = documents_dir or os.path.join(
            os.path.dirname(__file__), 
            "../../data/documents"
        )
        self.metadata_file = os.path.join(
            os.path.dirname(__file__),
            "../../data/documents_metadata.json"
        )
        
        # Database configuration
        self.use_db = use_db
        # For PostgreSQL, db_path is not used (network connection)
        self.db_path = db_path  # Kept for backward compatibility, not used
        
        self.batch_size = batch_size
        self.processed_docs: Dict[str, DocumentMetadata] = {}  # Only keep recent docs in memory
        self.hash_index: Set[str] = set()
        self._hash_cache: Dict[str, str] = {}
        self._db_conn: Optional[psycopg2.extensions.connection] = None
        self._db_lock = threading.Lock()
        self._processed_count = 0
        self._failed_count = 0
        
        Path(self.documents_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize database if needed
        if self.use_db:
            self._init_database()
        
        logger.info(f"DocumentLoader initialized with directory: {self.documents_dir}")
        logger.info(f"Database backend: {'PostgreSQL' if self.use_db else 'In-memory'}")
    
    def _init_database(self) -> None:
        """Initialize PostgreSQL database for metadata storage"""
        try:
            # Connect to PostgreSQL
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
            
            # Create documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    path TEXT UNIQUE,
                    file_type TEXT,
                    pages INTEGER,
                    file_size BIGINT,
                    hash TEXT UNIQUE,
                    loaded_at TEXT,
                    status TEXT,
                    content BYTEA,
                    content_compressed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create hash index for faster lookups
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hash_index (
                    hash TEXT PRIMARY KEY,
                    doc_id TEXT,
                    FOREIGN KEY (doc_id) REFERENCES documents(id)
                )
            """)
            
            # Create processing log for resume capability
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_log (
                    id SERIAL PRIMARY KEY,
                    file_path TEXT UNIQUE,
                    status TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash)")
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f"PostgreSQL database initialized: {POSTGRES_DB}@{POSTGRES_HOST}:{POSTGRES_PORT}")
        except psycopg2.Error as e:
            logger.error(f"Error initializing PostgreSQL database: {e}")
            self.use_db = False
        except Exception as e:
            logger.error(f"Unexpected error initializing database: {e}")
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
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Fast file hash using first+last chunks only"""
        if file_path in self._hash_cache:
            return self._hash_cache[file_path]
        
        try:
            file_size = os.path.getsize(file_path)
            
            # Skip very large files
            if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                logger.warning(f"File too large (>{MAX_FILE_SIZE_MB}MB): {file_path}")
                return ""
            
            # Fast hash: file size + first 32KB + last 32KB
            quick_hash = hashlib.md5()
            quick_hash.update(str(file_size).encode())
            
            with open(file_path, "rb") as f:
                quick_hash.update(f.read(32768))
                if file_size > 65536:
                    f.seek(max(0, file_size - 32768))
                    quick_hash.update(f.read(32768))
            
            result = quick_hash.hexdigest()
            self._hash_cache[file_path] = result
            return result
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            return ""
    
    def _check_duplicate_in_db(self, file_hash: str) -> bool:
        """Check if hash already exists in database"""
        if not self.use_db:
            return file_hash in self.hash_index
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM hash_index WHERE hash = %s", 
                (file_hash,)
            )
            exists = cursor.fetchone() is not None
            cursor.close()
            return exists
        except psycopg2.Error as e:
            logger.error(f"PostgreSQL error checking duplicate: {e}")
            return False
        except Exception as e:
            logger.error(f"Error checking duplicate: {e}")
            return False
    
    def _save_to_db(self, metadata: DocumentMetadata, content: str = None) -> bool:
        """Save document metadata and optional content to database"""
        if not self.use_db:
            return False
        
        try:
            with self._db_lock:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                
                # Compress content if provided
                content_data = None
                compressed = False
                if content:
                    content_bytes = content.encode('utf-8')
                    if len(content_bytes) > 1024:  # Only compress if > 1KB
                        content_data = psycopg2.Binary(gzip.compress(content_bytes))
                        compressed = True
                    else:
                        content_data = psycopg2.Binary(content_bytes)
                
                # PostgreSQL: Use ON CONFLICT for upsert
                cursor.execute("""
                    INSERT INTO documents 
                    (id, name, path, file_type, pages, file_size, hash, loaded_at, status, content, content_compressed)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        path = EXCLUDED.path,
                        file_type = EXCLUDED.file_type,
                        pages = EXCLUDED.pages,
                        file_size = EXCLUDED.file_size,
                        hash = EXCLUDED.hash,
                        loaded_at = EXCLUDED.loaded_at,
                        status = EXCLUDED.status,
                        content = EXCLUDED.content,
                        content_compressed = EXCLUDED.content_compressed
                """, (
                    metadata.id, metadata.name, metadata.path, metadata.file_type,
                    metadata.pages, metadata.file_size, metadata.hash,
                    metadata.loaded_at, metadata.status, content_data, compressed
                ))
                
                # Add to hash index (ON CONFLICT DO NOTHING for idempotency)
                cursor.execute("""
                    INSERT INTO hash_index (hash, doc_id)
                    VALUES (%s, %s)
                    ON CONFLICT (hash) DO NOTHING
                """, (metadata.hash, metadata.id))
                
                conn.commit()
                cursor.close()
                return True
        except psycopg2.extensions.QueryCanceledError as e:
            logger.error(f"PostgreSQL query timeout (table locked?): {e}")
            logger.error(f"HINT: Clear PostgreSQL tables manually with: DELETE FROM chunks; DELETE FROM hash_index; DELETE FROM documents; DELETE FROM processing_log;")
            if conn:
                conn.rollback()
            return False
        except psycopg2.Error as e:
            logger.error(f"PostgreSQL error saving to database: {e}")
            if conn:
                conn.rollback()
            return False
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            if conn:
                conn.rollback()
            return False
    
    def _load_pdf_fast(self, file_path: str) -> Tuple[str, int]:
        """Load PDF using PyMuPDF (10x faster than unstructured for PDFs)
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (text_content, page_count)
        """
        try:
            if not PYMUPDF_AVAILABLE:
                return None, 0  # Fall back to unstructured
            
            doc = fitz.open(file_path)
            file_name = os.path.basename(file_path)
            
            parts = [f"--- Document: {file_name} ---\n"]
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    parts.append(text)
            
            doc.close()
            content = "\n".join(parts)
            page_count = len(doc)
            
            return content, page_count
        except Exception as e:
            logger.debug(f"PyMuPDF failed for {file_path}, falling back to unstructured: {e}")
            return None, 0  # Fall back to unstructured
    
    def _load_document_with_unstructured(self, file_path: str) -> Tuple[str, int]:
        """Load document using unstructured library (supports many formats)
        
        For PDFs, tries PyMuPDF first (10x faster) then falls back to unstructured.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Tuple of (text_content, estimated_page_count)
        """
        try:
            # Fast path: Try PyMuPDF for PDFs (10x speed improvement)
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.pdf' and PYMUPDF_AVAILABLE:
                content, pages = self._load_pdf_fast(file_path)
                if content:  # Success
                    return content, pages
                # If None returned, fall through to unstructured
            
            # Standard path: Use unstructured for all formats (or PDF fallback)
            if not partition:
                logger.error("unstructured library not installed. Install with: pip install unstructured[pdf]")
                return "", 0
            
            # Suppress pandas DtypeWarning for CSV files with mixed types
            # The unstructured library internally uses pandas.read_csv which can emit these warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=Warning, message='.*mixed types.*')
                warnings.filterwarnings('ignore', category=Warning, message='.*Columns.*have mixed types.*')
                
                # Partition the document into elements
                elements = partition(filename=file_path)
            
            if not elements:
                logger.warning(f"No content extracted from: {file_path}")
                return "", 0
            
            # Combine all elements into single text
            file_name = os.path.basename(file_path)
            
            parts = [f"--- Document: {file_name} ---\n"]
            
            for element in elements:
                text = str(element).strip()
                if text:
                    parts.append(text)
            
            content = "\n".join(parts)
            
            # Estimate page count based on element count (rough estimate)
            estimated_pages = max(1, len(elements) // 10) if elements else 1
            
            return content, estimated_pages
        except Exception as e:
            logger.error(f"Error loading document: {file_path} - {str(e)}")
            return "", 0
    
    def _load_document(self, file_path: str, store_content: bool = False) -> Optional[DocumentMetadata]:
        """Load a single document file (supports multiple formats)
        
        Args:
            file_path: Path to document file
            store_content: Whether to store full content (use only if needed)
            
        Returns:
            DocumentMetadata object or None if loading failed
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None
            
            # Get file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Check if extension is supported
            if file_ext not in SUPPORTED_EXTENSIONS:
                return None
            
            # Get file size early
            file_size = os.path.getsize(file_path)
            
            if file_size == 0:
                logger.warning(f"Empty file: {file_path}")
                return None
            
            # Skip very large files
            if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                logger.warning(f"File too large (>{MAX_FILE_SIZE_MB}MB): {file_path}")
                return None
            
            # Calculate file hash for deduplication
            file_hash = self._calculate_file_hash(file_path)
            if not file_hash or self._check_duplicate_in_db(file_hash):
                if file_hash:
                    logger.debug(f"Document already loaded (duplicate): {file_path}")
                return None
            
            # Get file info
            file_name = os.path.basename(file_path)
            
            # Load document content - use direct CSV reading for CSV files to preserve structure
            if file_ext == '.csv':
                # Read CSV directly to preserve column structure for Q&A detection
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    num_pages = 1
                except Exception as e:
                    logger.error(f"Error reading CSV file {file_path}: {e}")
                    return None
            else:
                # Use unstructured for other file types
                content, num_pages = self._load_document_with_unstructured(file_path)
            
            if not content:
                logger.debug(f"No content extracted from: {file_path}")
                return None
            
            # Create metadata object (lightweight)
            doc_id = f"doc_{hashlib.md5(file_name.encode()).hexdigest()[:8]}"
            metadata = DocumentMetadata(
                id=doc_id,
                name=file_name,
                path=file_path,
                file_type=file_ext.lstrip('.'),
                pages=num_pages,
                file_size=file_size,
                hash=file_hash,
                loaded_at=datetime.now().isoformat(),
                status="loaded"
            )
            
            # Save to database if enabled
            if self.use_db:
                self._save_to_db(metadata, content if store_content else None)
            
            # Keep in memory for recent batch only
            self.processed_docs[doc_id] = metadata
            self.hash_index.add(file_hash)
            
            logger.debug(f"Successfully loaded {file_ext.upper()} file: {file_name}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            self._failed_count += 1
            return None
    
    def _discover_files(self) -> Generator[Path, None, None]:
        """Generator-based file discovery (memory efficient for millions of files)
        
        Yields:
            Path objects for supported files one at a time
        """
        logger.info(f"Starting file discovery in {self.documents_dir}")
        
        try:
            for ext in SUPPORTED_EXTENSIONS:
                # Use glob with generator to avoid loading all paths at once
                for file_path in Path(self.documents_dir).glob(f"**/*{ext}"):
                    yield file_path
        except Exception as e:
            logger.error(f"Error during file discovery: {e}")
    
    def load_all_documents(self, store_content: bool = False) -> List[DocumentMetadata]:
        """Load all documents using batch processing for scalability
        
        Supports: PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, JSON, XML, HTML, and more
        
        Args:
            store_content: Whether to store full content in database (uses more storage)
            
        Returns:
            List of loaded DocumentMetadata objects
        """
        logger.info(f"Starting document loading from {self.documents_dir}")
        logger.info(f"Batch size: {self.batch_size}, Workers: {MAX_WORKERS}")
        
        loaded_docs = []
        batch = []
        total_processed = 0
        
        # Generator-based file discovery
        file_discovery = self._discover_files()
        
        # Use ThreadPoolExecutor with futures for parallel processing
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures_to_file = {}
            
            # Process files in batches
            for file_path in file_discovery:
                batch.append(file_path)
                
                # Submit batch when it reaches batch size
                if len(batch) >= self.batch_size:
                    for file in batch:
                        future = executor.submit(self._load_document, str(file), store_content)
                        futures_to_file[future] = str(file)
                    batch = []
                    
                    # Process completed futures
                    for future in as_completed(futures_to_file):
                        try:
                            doc = future.result()
                            if doc:
                                loaded_docs.append(doc)
                                self._processed_count += 1
                        except Exception as e:
                            logger.error(f"Error loading {futures_to_file[future]}: {e}")
                            self._failed_count += 1
                        
                        del futures_to_file[future]
                        total_processed += 1
                        
                        # Log progress periodically
                        if total_processed % (self.batch_size * 10) == 0:
                            logger.info(f"Progress: {total_processed} files processed, {len(loaded_docs)} loaded")
                    
                    # Memory management: clear in-memory storage after batch
                    if len(self.processed_docs) > MEMORY_LIMIT_MB / 2:  # Simple estimate
                        self.processed_docs.clear()
            
            # Process remaining batch
            for file in batch:
                future = executor.submit(self._load_document, str(file), store_content)
                futures_to_file[future] = str(file)
            
            # Wait for all remaining futures
            for future in as_completed(futures_to_file):
                try:
                    doc = future.result()
                    if doc:
                        loaded_docs.append(doc)
                        self._processed_count += 1
                except Exception as e:
                    logger.error(f"Error loading {futures_to_file[future]}: {e}")
                    self._failed_count += 1
                
                total_processed += 1
        
        logger.info(f"Document loading complete: {len(loaded_docs)} loaded, {self._failed_count} failed")
        return loaded_docs
    
    def save_metadata(self) -> bool:
        """Save checkpoint metadata to JSON (for backward compatibility)
        
        Returns:
            True if successful, False otherwise
        """
        if self.use_db:
            logger.info("Metadata stored in database - no JSON export needed")
            return True
        
        try:
            metadata = {
                "total_documents": len(self.processed_docs),
                "last_updated": datetime.now().isoformat(),
                "documents": [asdict(doc) for doc in self.processed_docs.values()]
            }
            
            os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
            with open(self.metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Metadata saved to {self.metadata_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            return False

    def load_metadata(self) -> Optional[Dict]:
        """Load previously saved metadata
        
        Returns:
            Dictionary with metadata or None if not found
        """
        if self.use_db:
            try:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'loaded'")
                count = cursor.fetchone()[0]
                cursor.close()
                return {"total_documents": count, "source": "database"}
            except psycopg2.Error as e:
                logger.error(f"PostgreSQL error loading metadata: {e}")
                return None
            except Exception as e:
                logger.error(f"Error loading metadata from database: {e}")
                return None
        
        try:
            if not os.path.exists(self.metadata_file):
                return None
            
            with open(self.metadata_file, "r") as f:
                metadata = json.load(f)
            
            logger.info(f"Metadata loaded from {self.metadata_file}")
            return metadata
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            return None
    
    def get_documents(self, limit: int = None) -> List[DocumentMetadata]:
        """Get loaded documents (from recent batch in memory)
        
        For large scale, use query methods instead for database access
        
        Args:
            limit: Maximum number of documents to return
            
        Returns:
            List of DocumentMetadata objects
        """
        docs = list(self.processed_docs.values())
        if limit:
            return docs[:limit]
        return docs
    
    def get_documents_from_db(self, limit: int = 1000) -> List[Dict]:
        """Get documents directly from database (efficient for large scale)
        
        Args:
            limit: Maximum number of documents to retrieve
            
        Returns:
            List of document dictionaries
        """
        if not self.use_db:
            return []
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, name, path, file_type, pages, file_size, hash, loaded_at, status "
                "FROM documents WHERE status = 'loaded' LIMIT %s",
                (limit,)
            )
            
            docs = []
            for row in cursor.fetchall():
                docs.append({
                    "id": row[0], "name": row[1], "path": row[2], "file_type": row[3],
                    "pages": row[4], "file_size": row[5], "hash": row[6],
                    "loaded_at": row[7], "status": row[8]
                })
            
            cursor.close()
            return docs
        except psycopg2.Error as e:
            logger.error(f"PostgreSQL error querying database: {e}")
            return []
        except Exception as e:
            logger.error(f"Error querying database: {e}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        """Get a specific document by ID
        
        Args:
            doc_id: Document ID
            
        Returns:
            DocumentMetadata object or None if not found
        """
        return self.processed_docs.get(doc_id)
    
    def get_document_content(self, doc_id: str) -> Optional[str]:
        """Retrieve full document content from database
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document content or None if not found
        """
        if not self.use_db:
            return None
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT content, content_compressed FROM documents WHERE id = %s",
                (doc_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            
            if not row:
                return None
            
            content_data, compressed = row
            if content_data is None:
                return None
                
            # PostgreSQL returns memoryview or bytes for BYTEA fields
            if isinstance(content_data, memoryview):
                content_data = bytes(content_data)
            
            if compressed:
                return gzip.decompress(content_data).decode('utf-8')
            else:
                return content_data.decode('utf-8')
        except psycopg2.Error as e:
            logger.error(f"PostgreSQL error retrieving document content: {e}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving document content: {e}")
            return None
    
    def get_document_count(self) -> int:
        """Get total count of loaded documents
        
        Returns:
            Number of documents
        """
        if self.use_db:
            try:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM documents WHERE status = 'loaded'")
                count = cursor.fetchone()[0]
                cursor.close()
                return count
            except psycopg2.Error as e:
                logger.error(f"PostgreSQL error getting document count: {e}")
                return len(self.processed_docs)
            except Exception as e:
                logger.error(f"Error getting document count: {e}")
                return len(self.processed_docs)
        
        return len(self.processed_docs)
    
    def load_text_file(self, file_path: str) -> Optional[DocumentMetadata]:
        """Load a single text file (.txt) - Backward compatibility wrapper
        
        Args:
            file_path: Path to text file
            
        Returns:
            DocumentMetadata object or None if loading failed
        """
        return self._load_document(file_path, store_content=True)
    
    def load_pdf(self, file_path: str) -> Optional[DocumentMetadata]:
        """Load a single PDF file - Backward compatibility wrapper
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            DocumentMetadata object or None if loading failed
        """
        return self._load_document(file_path, store_content=True)
    
    def get_statistics(self) -> Dict:
        """Get statistics about loaded documents
        
        Returns:
            Dictionary with statistics (total_documents, total_pages, total_size_bytes, etc.)
        """
        if self.use_db:
            try:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*), SUM(pages), SUM(file_size), AVG(pages)
                    FROM documents WHERE status = 'loaded'
                """)
                row = cursor.fetchone()
                cursor.close()
                count, total_pages, total_size, avg_pages = row
                
                return {
                    "total_documents": count or 0,
                    "total_pages": int(total_pages) if total_pages else 0,
                    "total_size_bytes": int(total_size) if total_size else 0,
                    "average_pages_per_doc": float(avg_pages) if avg_pages else 0,
                    "documents_dir": self.documents_dir,
                    "backend": "PostgreSQL",
                    "failed_count": self._failed_count
                }
            except psycopg2.Error as e:
                logger.error(f"PostgreSQL error getting statistics: {e}")
            except Exception as e:
                logger.error(f"Error getting statistics: {e}")
        
        total_pages = sum(doc.pages for doc in self.processed_docs.values())
        total_size = sum(doc.file_size for doc in self.processed_docs.values())
        
        return {
            "total_documents": len(self.processed_docs),
            "total_pages": total_pages,
            "total_size_bytes": total_size,
            "average_pages_per_doc": total_pages / len(self.processed_docs) if self.processed_docs else 0,
            "documents_dir": self.documents_dir,
            "backend": "in-memory",
            "failed_count": self._failed_count
        }
    
    def close(self) -> None:
        """Close database connection"""
        if self._db_conn:
            self._db_conn.close()
            self._db_conn = None
            logger.info("Database connection closed")


def load_documents(documents_dir: str = None, use_db: bool = True, batch_size: int = BATCH_SIZE) -> List[DocumentMetadata]:
    """Quick function to load all documents with optimal settings
    
    Args:
        documents_dir: Path to documents directory
        use_db: Use database backend for scalability
        batch_size: Number of files per batch
        
    Returns:
        List of loaded DocumentMetadata objects
    """
    loader = DocumentLoader(documents_dir, use_db=use_db, batch_size=batch_size)
    try:
        return loader.load_all_documents()
    finally:
        loader.close()
