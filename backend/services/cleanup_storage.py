"""Manual storage cleanup utility - Run when storage is low.

Usage (from project root):
    python -m backend.services.cleanup_storage
"""

import os
import sys
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Ensure project root is on path so imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)  # Ensure 'data/' paths resolve correctly

from dotenv import load_dotenv
load_dotenv()

from backend.database.models import SessionLocal
from backend.services.session_service import get_session_manager
from src.utils.Logger import get_logger

logger = get_logger(__name__)


def check_disk_space():
    """Check and display current disk usage."""
    stat = shutil.disk_usage("data")
    total_gb = stat.total / (1024**3)
    used_gb = stat.used / (1024**3)
    free_gb = stat.free / (1024**3)
    usage_percent = (used_gb / total_gb) * 100
    
    print("\n" + "="*60)
    print("📊 DISK SPACE REPORT")
    print("="*60)
    print(f"Total Space:     {total_gb:.2f} GB")
    print(f"Used Space:      {used_gb:.2f} GB ({usage_percent:.1f}%)")
    print(f"Available Space: {free_gb:.2f} GB")
    print("="*60)
    
    if free_gb < 1.0:
        print("⚠️  CRITICAL: Less than 1GB free! Cleanup required.")
    elif free_gb < 2.0:
        print("⚠️  WARNING: Less than 2GB free. Consider cleanup.")
    elif free_gb < 5.0:
        print("ℹ️  INFO: Storage is getting low.")
    else:
        print("✅ Storage space is healthy.")
    
    return free_gb


def get_folder_size(folder: Path) -> float:
    """Get size of folder in GB."""
    if not folder.exists():
        return 0.0
    
    total = 0
    for item in folder.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    
    return total / (1024**3)


def analyze_storage():
    """Analyze what's using storage."""
    print("\n" + "="*60)
    print("📁 STORAGE BREAKDOWN")
    print("="*60)
    
    data_path = Path("data")
    
    folders = {
        "documents": data_path / "documents",
        "sessions": data_path / "sessions",
        "chunks": data_path / "chunks",
        "vector_store": data_path / "vector_store"
    }
    
    total_size = 0
    for name, folder in folders.items():
        size_gb = get_folder_size(folder)
        total_size += size_gb
        status = "📄" if size_gb < 0.1 else "📦" if size_gb < 1.0 else "📊"
        print(f"{status} {name:20s}: {size_gb:>8.2f} GB")
    
    print("-"*60)
    print(f"   {'TOTAL':20s}: {total_size:>8.2f} GB")
    print("="*60)


def cleanup_old_documents(days_old: int = 7):
    """Delete documents older than specified days."""
    docs_dir = Path("data/documents")
    if not docs_dir.exists():
        print("ℹ️  No documents directory found.")
        return
    
    cutoff = datetime.now() - timedelta(days=days_old)
    deleted_count = 0
    deleted_size = 0
    
    print(f"\n🗑️  Cleaning up documents older than {days_old} days...")
    
    for doc_file in docs_dir.glob("*"):
        if doc_file.is_file():
            file_time = datetime.fromtimestamp(doc_file.stat().st_mtime)
            if file_time < cutoff:
                file_size = doc_file.stat().st_size
                doc_file.unlink()
                deleted_count += 1
                deleted_size += file_size
                print(f"   Deleted: {doc_file.name}")
    
    deleted_size_mb = deleted_size / (1024**2)
    print(f"✅ Deleted {deleted_count} old documents ({deleted_size_mb:.2f} MB freed)")


def cleanup_all_sessions():
    """Delete all sessions (force cleanup)."""
    print("\n🗑️  Cleaning up ALL sessions...")
    
    db = SessionLocal()
    try:
        session_manager = get_session_manager()
        session_manager.cleanup_inactive_sessions(db)
        
        # Also clean any orphaned session directories
        sessions_dir = Path("data/sessions")
        if sessions_dir.exists():
            for session_dir in sessions_dir.iterdir():
                if session_dir.is_dir():
                    shutil.rmtree(session_dir)
                    print(f"   Removed: {session_dir.name}")
        
        print("✅ All sessions cleaned up")
    finally:
        db.close()


def main():
    """Main cleanup interface."""
    print("\n" + "="*60)
    print("🧹 RAG ASSISTANT - STORAGE CLEANUP UTILITY")
    print("="*60)
    
    # Check disk space
    free_gb = check_disk_space()
    
    # Analyze storage
    analyze_storage()
    
    # Menu
    print("\n" + "="*60)
    print("CLEANUP OPTIONS")
    print("="*60)
    print("1. Cleanup expired sessions (15+ min old OR 30+ min idle)")
    print("2. Cleanup old documents (7+ days)")
    print("3. Cleanup old documents (30+ days)")
    print("4. Force cleanup ALL sessions")
    print("5. Exit")
    print("="*60)
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == "1":
        print("\n🧹 Cleaning up inactive sessions...")
        db = SessionLocal()
        try:
            session_manager = get_session_manager()
            session_manager.cleanup_inactive_sessions(db)
            print("✅ Cleanup complete!")
        finally:
            db.close()
    
    elif choice == "2":
        cleanup_old_documents(days_old=7)
    
    elif choice == "3":
        cleanup_old_documents(days_old=30)
    
    elif choice == "4":
        confirm = input("⚠️  Delete ALL sessions? This cannot be undone! (yes/no): ").strip().lower()
        if confirm == "yes":
            cleanup_all_sessions()
        else:
            print("❌ Cancelled")
    
    elif choice == "5":
        print("\n👋 Goodbye!")
        return
    
    else:
        print("❌ Invalid option")
        return
    
    # Show updated disk space
    print("\n")
    check_disk_space()
    print()


if __name__ == "__main__":
    main()
