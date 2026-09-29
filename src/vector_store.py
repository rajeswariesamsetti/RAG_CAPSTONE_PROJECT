"""
ChromaDB Vector Store module for persistent embedding storage and retrieval.
Manages collection lifecycle: creation, chunk insertion, indexing, persistence, and safe deletion.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from src.config import CHROMA_DIR, CHROMA_COLLECTION_NAME
from src.embeddings import get_embedding_model
from utils.logging_utils import get_logger

logger = get_logger("vector_store")

def get_chroma_client(persist_dir: Path = CHROMA_DIR) -> chromadb.PersistentClient:
    """
    Returns a persistent ChromaDB client pointing to the chroma_db directory.
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(persist_dir.resolve()),
        settings=ChromaSettings(allow_reset=True, anonymized_telemetry=False)
    )

def get_chroma_collection(
    collection_name: str = CHROMA_COLLECTION_NAME,
    client: Optional[chromadb.PersistentClient] = None
):
    """
    Retrieves or creates the target ChromaDB collection.
    """
    if client is None:
        client = get_chroma_client()
    return client.get_or_create_collection(name=collection_name)

def get_vector_store(
    collection_name: str = CHROMA_COLLECTION_NAME,
    client: Optional[chromadb.PersistentClient] = None
) -> ChromaVectorStore:
    """
    Returns a LlamaIndex ChromaVectorStore wrapper around the ChromaDB collection.
    """
    collection = get_chroma_collection(collection_name=collection_name, client=client)
    return ChromaVectorStore(chroma_collection=collection)

def load_vector_index() -> Optional[VectorStoreIndex]:
    """
    Reloads an existing VectorStoreIndex from ChromaDB without re-computing embeddings.
    Returns None if the collection is empty.
    """
    client = get_chroma_client()
    collection = get_chroma_collection(client=client)

    if collection.count() == 0:
        logger.info("ChromaDB collection is empty. No existing index to load.")
        return None

    embed_model = get_embedding_model()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )
    logger.info(f"Loaded existing index with {collection.count()} chunks from ChromaDB.")
    return index

def build_vector_index(
    nodes: List[BaseNode],
    progress_callback=None
) -> VectorStoreIndex:
    """
    Indexes document nodes into ChromaDB and constructs the VectorStoreIndex.
    Overwrites/resets the current collection before re-indexing.

    Args:
        nodes: List of document chunks (BaseNode/TextNode).
        progress_callback: Optional callable(step_name, percent) for UI progress bars.

    Returns:
        VectorStoreIndex: The ready-to-query index.
    """
    if not nodes:
        raise ValueError("Cannot build vector index with zero document nodes.")

    if progress_callback:
        progress_callback("Initializing ChromaDB...", 10)

    client = get_chroma_client()
    # Reset existing collection cleanly
    try:
        client.delete_collection(name=CHROMA_COLLECTION_NAME)
        logger.info(f"Cleared existing collection '{CHROMA_COLLECTION_NAME}'.")
    except Exception:
        pass

    collection = client.create_collection(name=CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if progress_callback:
        progress_callback("Loading embedding model...", 30)

    embed_model = get_embedding_model()

    if progress_callback:
        progress_callback(f"Generating embeddings for {len(nodes)} chunks...", 50)

    logger.info(f"Generating embeddings and indexing {len(nodes)} chunks into ChromaDB...")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    if progress_callback:
        progress_callback("Knowledge base indexed successfully!", 100)

    logger.info(f"Successfully built knowledge base with {collection.count()} vectors.")
    return index

def clear_vector_database() -> bool:
    """
    Safely clears the ChromaDB collection.
    """
    client = get_chroma_client()
    try:
        client.delete_collection(name=CHROMA_COLLECTION_NAME)
        # Re-create empty collection
        client.create_collection(name=CHROMA_COLLECTION_NAME)
        logger.info("ChromaDB collection cleared successfully.")
        return True
    except Exception as e:
        logger.error(f"Error resetting ChromaDB collection: {e}")
        return False

def delete_document_vectors(filename: str) -> int:
    """
    Deletes all vectors associated with a specific document (by file_name metadata)
    from the ChromaDB collection.

    Args:
        filename: The file_name stored in chunk metadata.

    Returns:
        int: Number of vectors deleted.
    """
    client = get_chroma_client()
    collection = get_chroma_collection(client=client)

    try:
        results = collection.get(where={"file_name": filename}, include=[])
        ids = results.get("ids", [])

        if ids:
            collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} vectors for document: {filename}")

        return len(ids)
    except Exception as e:
        logger.error(f"Error deleting vectors for '{filename}': {e}")
        return 0


def get_collection_stats() -> Dict[str, Any]:
    """
    Returns metrics on the current state of the vector store:
      - chunk_count
      - is_empty
      - unique_documents
    """
    client = get_chroma_client()
    collection = get_chroma_collection(client=client)
    count = collection.count()

    unique_docs = set()
    if count > 0:
        try:
            # Sample metadata to find distinct files
            metadata_results = collection.get(include=["metadatas"])
            for meta in metadata_results.get("metadatas", []):
                if meta and "file_name" in meta:
                    unique_docs.add(meta["file_name"])
        except Exception as e:
            logger.warning(f"Could not extract unique documents from metadata: {e}")

    return {
        "chunk_count": count,
        "is_empty": count == 0,
        "document_count": len(unique_docs),
        "document_names": sorted(list(unique_docs)),
        "collection_name": CHROMA_COLLECTION_NAME
    }
