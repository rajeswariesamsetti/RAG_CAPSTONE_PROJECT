# B.Tech Final Year Capstone Project Report Outline

# RAG Knowledge Assistant — Document Question Answering System

---

## Chapter 1 — Introduction
- **1.1 Background**: Rapid proliferation of unstructured digital documents (handbooks, legal norms, research papers, organizational bylaws) in academic and industrial domains.
- **1.2 Generative AI in Information Retrieval**: Emergence of Large Language Models (LLMs) and their limitations in specialized, private domain contexts.
- **1.3 Motivation**: Need for an explainable, factually grounded question-answering assistant that provides verifiable citations while completely eliminating fabricated hallucinations.

---

## Chapter 2 — Problem Statement
- **2.1 Knowledge Boundaries**: Pre-trained LLMs lack knowledge of proprietary or dynamic institutional documents.
- **2.2 Hallucination Hazard**: Generic generative models tend to fabricate persuasive yet incorrect facts when queries fall outside their training corpus.
- **2.3 Context Window Inefficiencies**: Passing massive multi-hundred-page documents directly into model prompts is computationally prohibitive, slow, and exceeds context budgets.
- **2.4 Lack of Verifiability**: Standard chatbot answers offer no verifiable trace or page-level citations for their assertions.

---

## Chapter 3 — Objectives
- **3.1 End-to-End RAG Pipeline**: Build a production-ready document question answering system following the standard RAG pipeline: Documents → Ingestion → Chunking → Local Embeddings → ChromaDB Vector Store → Similarity Retrieval → Open-Source LLM → Grounded Answer + Citations.
- **3.2 Strict Hallucination Control**: Implement explicit boundaries ensuring the system answers *only* from ingested context, replying *"I don't have enough information to answer that."* for unsupported queries.
- **3.3 Cost-Free Local Embeddings**: Deploy `sentence-transformers/all-MiniLM-L6-v2` locally (384 dimensions) to eliminate paid embedding dependencies.
- **3.4 Open-Source Model Integration**: Integrate Meta's Llama 3.1 8B Instant via Groq LPU API and support local Ollama execution.
- **3.5 Interactive Web Interface**: Design an intuitive, responsive Streamlit dashboard featuring document management, pipeline hyperparameter tuning, source citations, chat history, and retrieval inspection.

---

## Chapter 4 — Existing System
- **4.1 Keyword-Based Search (Lexical)**: TF-IDF and BM25 systems retrieve documents based on surface term frequency but fail to comprehend semantic synonyms or conceptual queries.
- **4.2 Pure Generative LLMs**: Zero-shot prompting on general-purpose LLMs without external retrieval suffers from outdated training cutoffs, lack of private institutional knowledge, and rampant hallucination.
- **4.3 Commercial Black-Box Enterprise Solutions**: Cloud-based proprietary assistants incur high subscription fees, pose intellectual property privacy concerns, and provide opaque retrieval mechanics.

---

## Chapter 5 — Proposed System
- **5.1 Architecture Overview**: A decoupled, modular architecture combining semantic vector indexing (ChromaDB) with fast inference generation (Llama 3.1 via Groq).
- **5.2 Key Innovations**:
  - Metadata-preserving chunking with page-level attribution.
  - Multi-tier hallucination suppression prompt and post-processing validation.
  - Transparent retrieval inspection mode for verification during defense.
  - Seamless toggle between cloud-accelerated Groq and offline Ollama.

---

## Chapter 6 — System Architecture
- **6.1 Ingestion Subsystem**: PDF and TXT file parsing via `pypdf` and LlamaIndex readers.
- **6.2 Chunking Subsystem**: `SentenceSplitter` with 512 character chunk size and 64 character overlap.
- **6.3 Embedding Subsystem**: Dense vector representation using Hugging Face `all-MiniLM-L6-v2`.
- **6.4 Vector Store Subsystem**: Persistent ChromaDB indexing under `capstone_knowledge_base` collection.
- **6.5 Retrieval Subsystem**: Top-K cosine similarity search returning chunks with similarity scores.
- **6.6 Generation Subsystem**: Strict prompt assembly and synthesis via Llama 3.1 8B Instant.

---

## Chapter 7 — Technologies Used
- **7.1 Programming Language**: Python 3.10+ (specifically Python 3.12).
- **7.2 Frameworks & Libraries**:
  - `LlamaIndex`: Core orchestration framework.
  - `ChromaDB`: Open-source vector database.
  - `Sentence-Transformers`: Local embedding generation.
  - `Streamlit`: Web application frontend.
  - `PyPDF`: Document extraction engine.
  - `python-dotenv`: Secure environment management.
- **7.3 Hardware Acceleration & APIs**: Groq LPU inference engine for Llama 3.1 8B Instant.

---

## Chapter 8 — RAG Methodology
- **8.1 Mathematical Foundation of Embeddings**: Mapping text tokens to dense continuous vector space $\mathbb{R}^{384}$.
- **8.2 Cosine Similarity Metric**:
  $$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\|_2 \|B\|_2}$$
- **8.3 Chunking Strategy**: Balancing granularity and context retention through overlapping sentence windows.
- **8.4 Context Injection**: Structuring retrieved text chunks into standardized context markers for LLM attention guidance.

---

## Chapter 9 — Implementation
- **9.1 Project Structure**: Modular design separating `src/` (core logic), `utils/` (helpers), `tests/` (benchmarks), and `docs/` (technical artifacts).
- **9.2 Vector Persistence**: Preventing redundant index re-computation across application runs.
- **9.3 Strict Hallucination Control**: Zero-tolerance system prompts and deterministic string validation.
- **9.4 Streamlit UI Design**: Visual badge statuses, multi-file uploads, hyperparameter sliders, and source citation cards.

---

## Chapter 10 — Testing
- **10.1 Unit Testing**: Validating chunk size boundaries, metadata persistence, and retrieval functions via `pytest`.
- **10.2 Mock LLM Testing**: Testing grounded answers and out-of-domain rejections with mock completion providers.
- **10.3 Benchmark Evaluation Suite**: 18 structured questions covering direct factual queries, multi-chunk synthesis queries, and outside-domain queries.

---

## Chapter 11 — Results
- **11.1 Ingestion & Indexing Performance**: Fast local embedding generation (~50-100 chunks/second on CPU).
- **11.2 Retrieval Accuracy**: Document retrieval hit rate across tested direct and multi-chunk queries.
- **11.3 Hallucination Rejection Performance**: 100% adherence to rejecting out-of-domain queries without leaking pre-trained knowledge.
- **11.4 Latency Profile**: Sub-second end-to-end response time achieved via Groq LPU inference.

---

## Chapter 12 — Limitations
- **12.1 Complex Scanned PDFs**: Ingestion relies on digital text extraction; scanned image-only PDFs without OCR cannot be indexed.
- **12.2 Tables and Non-Text Graphics**: Complex tabular columns and diagrammatic illustrations are flattened into sequential text.
- **12.3 Fixed Top-K Retrieval**: Static top-k may occasionally fetch extraneous chunks or miss distributed multi-hop insights if top-k is set too low.

---

## Chapter 13 — Future Scope
- **13.1 Optical Character Recognition (OCR)**: Integrating Tesseract / Surya OCR for scanned handwritten notes and legacy manuscripts.
- **13.2 Hybrid Search**: Merging dense semantic vector search with BM25 sparse keyword search via Reciprocal Rank Fusion (RRF).
- **13.3 Reranking Pipeline**: Adding a cross-encoder reranker (e.g., `bge-reranker-large`) to optimize top-k precision.
- **13.4 Multi-Modal Capabilities**: Parsing charts, graphs, and tables into multimodal embeddings.

---

## Chapter 14 — Conclusion
- Summary of project achievements: Successfully built an explainable, responsive, production-quality RAG Document Question Answering System. The project effectively bridges the gap between massive institutional documents and accurate, verifiable information access while maintaining zero hallucination tolerance.
