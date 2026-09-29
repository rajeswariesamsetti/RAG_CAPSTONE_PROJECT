# System Architecture & Technical Specifications

## RAG Knowledge Assistant — Document Question Answering System

This document outlines the architectural blueprint, data flow, component interactions, and engineering decisions implemented in the **RAG Knowledge Assistant**.

---

## 1. High-Level Architecture Diagram

```
                 OFFLINE / INGESTION PHASE
        ┌────────────────────────────────────────┐
        │       PDF / TXT Document Files         │
        └───────────────────┬────────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │            Document Loader             │
        │  (Extracts text, pages & file metadata)│
        └───────────────────┬────────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │            SentenceSplitter            │
        │   (chunk_size=512, chunk_overlap=64)   │
        └───────────────────┬────────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │       Local Embedding Model            │
        │ (sentence-transformers/all-MiniLM-L6-v2)│
        │           [384 Dimensions]             │
        └───────────────────┬────────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │       ChromaDB Vector Database         │
        │  (collection: capstone_knowledge_base) │
        └───────────────────┬────────────────────┘
                            │
════════════════════════════╪══════════════════════════════════════════════
                 ONLINE / QUERY PHASE
                            │
        ┌───────────────────┴────────────────────┐
        │ User Query via Streamlit Interface     │
        └───────────────────┬────────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │ Query Embedding Generation             │
        │ (all-MiniLM-L6-v2: Query Vector)       │
        └───────────────────┬────────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │ Cosine Similarity Retrieval            │
        │ (ChromaDB: Top-K = 3 closest chunks)   │
        └───────────────────┬────────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │ Retrieved Context Assembly             │
        │ (Source, Page, Chunk Text, Score)      │
        └───────────────────┬────────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │ Strict Prompt Construction             │
        │ (Boundary Enforcement, Zero Prior Know)│
        └───────────────────┬────────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │ Large Language Model                   │
        │ (Llama 3.1 8B Instant via Groq LPU API)│
        └───────────────────┬────────────────────┘
                            │
                            ▼
        ┌────────────────────────────────────────┐
        │ Grounded Answer + Source Citations     │
        │ (Or rejection: 'I don't have enough...')│
        └────────────────────────────────────────┘
```

---

## 2. Pipeline Subsystems

### 2.1 Document Ingestion Subsystem (`src/document_loader.py`)
- **Supported Formats**: `.pdf` (Portable Document Format) and `.txt` (Plain Text).
- **Metadata Extraction**:
  - `file_name`: Basename of the file.
  - `file_type`: Extension identifier.
  - `page_label`: Preserves original page numbers for PDF documents, ensuring accurate source tracing during citations.
- **Fail-safe Design**: Individual file read errors are captured without terminating the entire ingestion batch.

### 2.2 Semantic Chunking Subsystem (`src/chunking.py`)
- **Algorithm**: `SentenceSplitter` from LlamaIndex.
- **Default Hyperparameters**:
  - `chunk_size = 512` characters/tokens.
  - `chunk_overlap = 64` characters/tokens.
- **Metadata Inheritance**: Every text node inherits the originating document name, page label, and receives a unique `chunk_id` and sequential `chunk_index`. Overlapping guarantees context continuity across sentence boundaries.

### 2.3 Embedding Subsystem (`src/embeddings.py`)
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`.
- **Dimensions**: 384-dimensional dense semantic vectors.
- **Execution**: Runs 100% locally on the host machine using PyTorch and Hugging Face Transformers. No embedding API costs, data privacy preserved.
- **Normalization**: Vectors are normalized to unit length, enabling high-performance cosine similarity via dot product.

### 2.4 Vector Storage & Indexing Subsystem (`src/vector_store.py`)
- **Database Engine**: ChromaDB (persistent local storage at `./chroma_db`).
- **Collection Name**: `capstone_knowledge_base`.
- **Lifecycle Management**:
  - Reload existing database on app launch (zero redundant computation).
  - Clean rebuild upon user command.
  - Safe reset and cleanup options.

### 2.5 Retrieval Subsystem (`src/retriever.py`)
- **Metric**: Cosine Similarity.
- **Hyperparameter**: `similarity_top_k = 3` (configurable from 1 to 8).
- **Inspection Attributes**:
  - Source file name.
  - Page number (if available).
  - Cosine similarity score (0.0 to 1.0).
  - Exact snippet text.

### 2.6 LLM & Generation Subsystem (`src/llm.py` & `src/rag_pipeline.py`)
- **Default Provider**: Groq API.
- **Model**: `llama-3.1-8b-instant` (Meta's state-of-the-art open-weights model served on Groq's Language Processing Units).
- **Alternative Provider**: Ollama local daemon (`llama3.1:8b`).
- **Temperature**: `0.1` (low temperature to minimize creativity and strictly adhere to retrieved facts).

---

## 3. Hallucination Mitigation Architecture

The system incorporates a **three-tier defense** against hallucination:

1. **System Prompt Constraint**:
   Instructs the model that it is solely a retrieval synthesizer and has no access to external memory or world facts.
2. **Strict QA Prompt Template**:
   Instructs the model: *"If the answer cannot be found in the context above, respond exactly with: I don't have enough information to answer that."*
3. **Deterministic Output Normalization**:
   The `rag_pipeline.py` layer intercepts the generated text. If any standard lack-of-information phrasing is detected, it returns the standard canonical rejection:
   `"I don't have enough information to answer that."`

---

## 4. Source Citation Mechanism

Every retrieved response formats each context segment with explicit source tags:
```
[Source 1: Document: college_handbook.pdf | Page 2]
The attendance requirement is 75%...
```
The user interface parses the retrieved nodes and displays a dedicated citation card for each source, providing transparency, academic integrity, and auditability.
