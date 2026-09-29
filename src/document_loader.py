"""
Document Loader module using LlamaIndex.
Responsible for ingesting PDF and TXT files while extracting and preserving page/file metadata.
"""

from pathlib import Path
from typing import List, Union
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from src.config import DATA_DIR, SUPPORTED_EXTENSIONS
from utils.logging_utils import get_logger

logger = get_logger("document_loader")

def load_documents(
    input_files_or_dir: Union[str, Path, List[Union[str, Path]]] = DATA_DIR
) -> List[Document]:
    """
    Loads PDF and TXT documents using LlamaIndex readers.
    Preserves document metadata including:
      - file_name
      - file_type
      - page_label (for PDFs)

    Args:
        input_files_or_dir: A directory path, a single file path, or a list of file paths.

    Returns:
        List[Document]: List of LlamaIndex Document objects.
    """
    file_paths: List[Path] = []

    if isinstance(input_files_or_dir, (str, Path)):
        p = Path(input_files_or_dir)
        if p.is_dir():
            for ext in SUPPORTED_EXTENSIONS:
                file_paths.extend(p.glob(f"*{ext}"))
                file_paths.extend(p.glob(f"*{ext.upper()}"))
        elif p.is_file():
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                file_paths.append(p)
    elif isinstance(input_files_or_dir, list):
        for item in input_files_or_dir:
            p = Path(item)
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                file_paths.append(p)

    if not file_paths:
        logger.warning("No supported documents found to load.")
        return []

    logger.info(f"Loading {len(file_paths)} document(s): {[f.name for f in file_paths]}")

    try:
        reader = SimpleDirectoryReader(
            input_files=[str(fp.resolve()) for fp in file_paths],
            filename_as_id=True,
            raise_on_error=False
        )
        documents = reader.load_data()
    except Exception as e:
        logger.error(f"Error while loading documents with SimpleDirectoryReader: {e}")
        # Fallback: attempt manual plain text or per-file loading
        documents = []
        for fp in file_paths:
            try:
                single_reader = SimpleDirectoryReader(input_files=[str(fp.resolve())])
                docs = single_reader.load_data()
                documents.extend(docs)
            except Exception as single_err:
                logger.error(f"Failed to load file {fp.name}: {single_err}")

    # Standardize metadata across documents
    for doc in documents:
        # Ensure file_name is clean
        file_path_str = doc.metadata.get("file_path", "")
        if file_path_str:
            doc.metadata["file_name"] = Path(file_path_str).name
            doc.metadata["file_type"] = Path(file_path_str).suffix.lower()
        elif "file_name" not in doc.metadata:
            doc.metadata["file_name"] = "Unknown"
            doc.metadata["file_type"] = ".txt"

        # If page_label is not present, check for page number
        if "page_label" not in doc.metadata:
            if "page" in doc.metadata:
                doc.metadata["page_label"] = str(doc.metadata["page"])

    logger.info(f"Successfully loaded {len(documents)} page/document section(s).")
    return documents
