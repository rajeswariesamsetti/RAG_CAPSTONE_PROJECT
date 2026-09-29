"""
Unit tests for semantic similarity retrieval.
Verifies chunk retrieval, scoring, top-k limits, and empty query handling.
"""

import pytest
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.schema import TextNode
from src.retriever import retrieve_chunks
from src.embeddings import get_embedding_model

@pytest.fixture(scope="module")
def sample_index():
    """Builds a temporary lightweight in-memory vector index for retrieval testing."""
    embed_model = get_embedding_model()
    nodes = [
        TextNode(
            text="The attendance requirement for undergraduate programs is 75% in every course.",
            metadata={"file_name": "college_handbook.txt", "page_label": "1", "chunk_id": "c1"}
        ),
        TextNode(
            text="The Central Library is open from 8:00 AM to 10:00 PM on all working days.",
            metadata={"file_name": "college_handbook.txt", "page_label": "2", "chunk_id": "c2"}
        ),
        TextNode(
            text="Undergraduate B.Tech students must complete a total of 160 credits across eight semesters.",
            metadata={"file_name": "academic_rules.txt", "page_label": "1", "chunk_id": "c3"}
        ),
    ]
    return VectorStoreIndex(nodes=nodes, embed_model=embed_model)

def test_retrieve_empty_query(sample_index):
    """Empty query should return empty list without error."""
    results = retrieve_chunks("", sample_index, similarity_top_k=2)
    assert results == []

def test_retrieve_attendance_query(sample_index):
    """Query about attendance should return the attendance node as top result."""
    results = retrieve_chunks("What is the attendance requirement?", sample_index, similarity_top_k=2)
    assert len(results) > 0
    assert len(results) <= 2

    top = results[0]
    assert "attendance" in top["text"].lower()
    assert top["filename"] == "college_handbook.txt"
    assert top["page"] == "1"
    assert top["score"] is not None
    assert isinstance(top["score"], float)

def test_retrieve_credit_query(sample_index):
    """Query about credits should retrieve academic_rules.txt chunk."""
    results = retrieve_chunks("How many credits must a student complete?", sample_index, similarity_top_k=1)
    assert len(results) == 1
    assert "160 credits" in results[0]["text"]
    assert results[0]["filename"] == "academic_rules.txt"
