"""
Unit tests for document chunking module.
Verifies chunk splitting behavior, size constraints, and metadata preservation.
"""

import pytest
from llama_index.core import Document
from src.chunking import get_chunker, chunk_documents
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

def test_chunker_initialization():
    """Verify that SentenceSplitter is initialized with expected defaults."""
    chunker = get_chunker(chunk_size=512, chunk_overlap=64)
    assert chunker.chunk_size == 512
    assert chunker.chunk_overlap == 64

def test_chunk_documents_empty():
    """Verify handling of empty document list."""
    nodes = chunk_documents([])
    assert nodes == []

def test_chunk_documents_metadata_preservation():
    """Verify that chunks inherit and preserve originating metadata."""
    sample_text = (
        "The attendance requirement for all undergraduate programs is 75% in every "
        "registered theory and laboratory course. Students with attendance below 75% "
        "will not be permitted to appear for the end-semester examinations unless an "
        "official medical leave application approved by the Dean of Student Affairs "
        "has been submitted within 5 working days of absence.\n\n"
        "To be eligible for degree conferment, a student must secure a minimum cumulative "
        "grade point average (CGPA) of 5.0."
    )
    doc = Document(
        text=sample_text,
        metadata={
            "file_name": "college_handbook.pdf",
            "file_type": ".pdf",
            "page_label": "2"
        }
    )

    nodes = chunk_documents([doc], chunk_size=100, chunk_overlap=20)
    assert len(nodes) > 0

    for i, node in enumerate(nodes):
        assert node.metadata.get("file_name") == "college_handbook.pdf"
        assert node.metadata.get("page_label") == "2"
        assert node.metadata.get("chunk_index") == i
        assert "chunk_id" in node.metadata
        assert len(node.get_content()) > 0
