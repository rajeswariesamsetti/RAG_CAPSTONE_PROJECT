"""
Unit and integration tests for the RAG Pipeline and Hallucination Control.
Verifies grounded answer generation, source citation structures, and outside-knowledge rejection.
"""

import pytest
from unittest.mock import MagicMock
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from src.rag_pipeline import RAGPipeline
from src.config import OUT_OF_CONTEXT_RESPONSE
from src.embeddings import get_embedding_model

@pytest.fixture(scope="module")
def rag_test_index():
    """Builds a deterministic in-memory vector index for pipeline tests."""
    embed_model = get_embedding_model()
    nodes = [
        TextNode(
            text="The attendance requirement for all undergraduate programs is 75%.",
            metadata={"file_name": "college_handbook.txt", "page_label": "2"}
        ),
        TextNode(
            text="The Central Library is open on all working days from 8:00 AM to 10:00 PM.",
            metadata={"file_name": "college_handbook.txt", "page_label": "4"}
        )
    ]
    return VectorStoreIndex(nodes=nodes, embed_model=embed_model)

def test_rag_empty_question(rag_test_index):
    """Empty query should prompt user to ask a question."""
    pipeline = RAGPipeline(index=rag_test_index)
    res = pipeline.answer_question("")
    assert "Please ask a question" in res["answer"]
    assert res["sources"] == []

def test_rag_missing_index():
    """Pipeline without index should provide helpful guidance."""
    pipeline = RAGPipeline(index=None)
    res = pipeline.answer_question("What is attendance?")
    assert "No knowledge base loaded" in res["answer"]

def test_rag_grounded_answer(rag_test_index):
    """Test grounded answer synthesis with a mock LLM."""
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "The attendance requirement is 75%."

    pipeline = RAGPipeline(index=rag_test_index, llm=mock_llm)
    res = pipeline.answer_question("What is the attendance requirement?")

    assert "75%" in res["answer"]
    assert len(res["sources"]) > 0
    assert res["sources"][0]["filename"] == "college_handbook.txt"
    assert res["sources"][0]["page"] == "2"
    assert res["sources"][0]["score"] is not None

def test_rag_hallucination_rejection(rag_test_index):
    """
    Test that when the LLM signals lack of context or when outside questions are asked,
    the pipeline strictly outputs OUT_OF_CONTEXT_RESPONSE.
    """
    mock_llm = MagicMock()
    mock_llm.complete.return_value = "I don't have enough information to answer that."

    pipeline = RAGPipeline(index=rag_test_index, llm=mock_llm)
    res = pipeline.answer_question("Who won the FIFA World Cup?")

    assert res["answer"] == OUT_OF_CONTEXT_RESPONSE
