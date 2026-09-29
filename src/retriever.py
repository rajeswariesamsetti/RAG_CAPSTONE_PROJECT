"""
Retrieval module for semantic similarity search over ChromaDB embeddings.
Retrieves top-k relevant chunks with similarity scores and originating metadata.
"""

from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from src.config import DEFAULT_SIMILARITY_TOP_K
from utils.logging_utils import get_logger

logger = get_logger("retriever")

def get_retriever(index: VectorStoreIndex, similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K):
    """
    Constructs a LlamaIndex vector retriever with specified top-k parameter.
    """
    return index.as_retriever(similarity_top_k=similarity_top_k)

def retrieve_chunks(
    query: str,
    index: VectorStoreIndex,
    similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K
) -> List[Dict[str, Any]]:
    """
    Performs semantic similarity retrieval for a given query against the vector index.

    Args:
        query: User question or search query string.
        index: VectorStoreIndex connected to ChromaDB.
        similarity_top_k: Number of chunks to retrieve (default: 3).

    Returns:
        List[Dict[str, Any]]: Top-k retrieved chunks, each containing:
            - text: Text content of the chunk
            - filename: Originating source document filename
            - page: Page number/label if available (None otherwise)
            - score: Semantic similarity score (rounded to 4 decimal places)
            - chunk_id: Unique chunk identifier
            - raw_node: Underlying NodeWithScore object for LlamaIndex query synthesis
    """
    if not query or not query.strip():
        logger.warning("Empty query passed to retriever.")
        return []

    retriever = get_retriever(index, similarity_top_k=similarity_top_k)
    nodes_with_scores: List[NodeWithScore] = retriever.retrieve(query)

    retrieved_results: List[Dict[str, Any]] = []

    for item in nodes_with_scores:
        node = item.node
        score = round(float(item.score), 4) if item.score is not None else None

        filename = node.metadata.get("file_name") or node.metadata.get("filename") or "Unknown Document"
        page = node.metadata.get("page_label") or node.metadata.get("page") or None

        retrieved_results.append({
            "text": node.get_content(),
            "filename": filename,
            "page": page,
            "score": score,
            "chunk_id": node.metadata.get("chunk_id", node.node_id),
            "raw_node": item
        })

    logger.info(f"Retrieved {len(retrieved_results)} chunks for query: '{query[:40]}...'")
    return retrieved_results
