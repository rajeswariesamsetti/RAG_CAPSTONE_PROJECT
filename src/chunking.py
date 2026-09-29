"""
Chunking module using LlamaIndex SentenceSplitter.
Splits ingested documents into semantic text chunks while preserving originating metadata.
"""

from typing import List, Optional
from llama_index.core import Document
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.node_parser import SentenceSplitter
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from utils.logging_utils import get_logger

logger = get_logger("chunking")

def get_chunker(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> SentenceSplitter:
    """
    Returns a configured SentenceSplitter instance.
    """
    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        include_metadata=True,
        include_prev_next_rel=True
    )

def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[BaseNode]:
    """
    Splits a list of Documents into text chunks (Nodes) with metadata preservation.

    Args:
        documents: List of LlamaIndex Document instances.
        chunk_size: Size of chunks in tokens/characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List[BaseNode]: Processed nodes ready for vector embedding.
    """
    if not documents:
        logger.warning("No documents provided for chunking.")
        return []

    splitter = get_chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)

    # Enrich metadata for each chunk
    for idx, node in enumerate(nodes):
        node.metadata["chunk_index"] = idx
        node.metadata["chunk_id"] = node.node_id
        # Ensure file_name and page_label are accessible
        if "file_name" not in node.metadata:
            node.metadata["file_name"] = node.metadata.get("filename", "Unknown")
        if "page_label" not in node.metadata and "page" in node.metadata:
            node.metadata["page_label"] = str(node.metadata["page"])

    logger.info(
        f"Generated {len(nodes)} chunks from {len(documents)} document sections "
        f"(chunk_size={chunk_size}, chunk_overlap={chunk_overlap})."
    )
    return nodes
