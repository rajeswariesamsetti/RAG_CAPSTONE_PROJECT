"""
File management utilities for handling uploaded documents and data directory operations.
"""

import shutil
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from src.config import DATA_DIR, SUPPORTED_EXTENSIONS
from utils.logging_utils import get_logger

logger = get_logger("file_utils")

def is_supported_file(file_path: Union[str, Path]) -> bool:
    """Checks whether the file has a supported extension (.pdf, .txt)."""
    path = Path(file_path)
    return path.suffix.lower() in SUPPORTED_EXTENSIONS

def save_uploaded_file(uploaded_file, destination_dir: Path = DATA_DIR) -> Path:
    """
    Saves a Streamlit UploadedFile object into the destination directory.
    Returns the Path to the saved file.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    file_path = destination_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logger.info(f"Saved uploaded file: {file_path.name} ({file_path.stat().st_size} bytes)")
    return file_path

def get_uploaded_files(directory: Path = DATA_DIR) -> List[Dict[str, Any]]:
    """
    Lists all supported files in the data directory with metadata.
    """
    files = []
    if not directory.exists():
        return files

    for file_path in directory.iterdir():
        if file_path.is_file() and is_supported_file(file_path):
            files.append({
                "name": file_path.name,
                "size_kb": round(file_path.stat().st_size / 1024, 2),
                "extension": file_path.suffix.lower()
            })
    return sorted(files, key=lambda x: x["name"].lower())

def clear_data_directory(directory: Path = DATA_DIR) -> int:
    """
    Removes all uploaded documents from the data directory.
    Preserves .gitkeep if present.
    Returns the number of deleted files.
    """
    count = 0
    if not directory.exists():
        return count

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.name != ".gitkeep":
            try:
                file_path.unlink()
                count += 1
            except Exception as e:
                logger.error(f"Error deleting file {file_path.name}: {e}")
    logger.info(f"Cleared {count} files from {directory}")
    return count


def read_file_content(filename: str, data_dir: Path = DATA_DIR) -> Optional[str]:
    """
    Read text content from a file in the data directory.
    For TXT files, reads raw text. For PDFs, extracts text using pypdf.
    Returns None if the file cannot be read.
    """
    filepath = data_dir / filename

    # Security: ensure file stays within data directory
    try:
        filepath.resolve().relative_to(data_dir.resolve())
    except ValueError:
        logger.error(f"Security: path traversal attempt blocked for: {filename}")
        return None

    if not filepath.exists() or not filepath.is_file():
        return None

    ext = filepath.suffix.lower()

    if ext == ".txt":
        try:
            return filepath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Error reading TXT file {filename}: {e}")
            return None

    elif ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(filepath))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n".join(pages) if pages else None
        except Exception as e:
            logger.error(f"Error reading PDF file {filename}: {e}")
            return None

    return None


def delete_single_file(filename: str, data_dir: Path = DATA_DIR) -> bool:
    """
    Safely delete a specific file from the data directory.
    Validates the path to prevent directory traversal attacks.
    """
    filepath = data_dir / filename

    # Security: ensure file stays within data directory
    try:
        filepath.resolve().relative_to(data_dir.resolve())
    except ValueError:
        logger.error(f"Security: attempted to delete file outside data directory: {filename}")
        return False

    if filepath.exists() and filepath.is_file():
        try:
            filepath.unlink()
            logger.info(f"Deleted file: {filename}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file {filename}: {e}")
            return False
    return False

