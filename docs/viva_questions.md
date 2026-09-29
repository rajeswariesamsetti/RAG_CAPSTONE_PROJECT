# B.Tech Capstone Project — Viva Examination Preparation Guide

## RAG Knowledge Assistant: Document Question Answering System

This document contains 32 comprehensive questions and answers directly aligned with the architecture, implementation, and evaluation of this Capstone Project.

---

### Q1: What is RAG (Retrieval-Augmented Generation)?
**Answer**:
RAG is an AI architectural pattern that enhances the output of a Large Language Model by referencing an authoritative external knowledge base outside its training data before generating a response. It consists of two stages:
1. **Retrieval**: Querying a vector database to fetch relevant text chunks matching the user's question.
2. **Generation**: Passing the retrieved context into the LLM prompt to generate a grounded, fact-checked answer.

---

### Q2: Why use RAG instead of fine-tuning the LLM?
**Answer**:
1. **Cost & Computation**: Fine-tuning requires expensive GPU clusters and extensive training time, whereas RAG dynamically updates in seconds by simply adding documents to ChromaDB.
2. **Hallucination Control**: Fine-tuned models still hallucinate, while RAG forces the LLM to ground its answers strictly in retrieved passages.
3. **Verifiability**: RAG provides transparent, verifiable source citations (filename, page number, similarity score) for every claim.
4. **Data Privacy**: Sensitive organizational documents remain safely stored in local vector databases rather than baked into model weights.

---

### Q3: What is "Hallucination" in LLMs?
**Answer**:
Hallucination refers to a phenomenon where an LLM generates grammatically fluent, confident, and plausible-sounding statements that are factually false, ungrounded, or completely fabricated from outside the provided context.

---

### Q4: How does our project prevent hallucination?
**Answer**:
Our system employs a multi-tiered defense:
1. **System Prompt Constraint**: Sets strict operational boundaries, instructing the LLM that it has zero prior memory and must use *only* the injected context.
2. **Strict QA Prompt Template**: Instructs the LLM: *"If the answer cannot be found in the context above, respond exactly with: I don't have enough information to answer that."*
3. **Low Temperature (0.1)**: Minimizes creative sampling randomness, forcing deterministic, fact-focused token generation.
4. **Post-Processing Filter**: In `rag_pipeline.py`, any output indicating missing information is normalized to the exact canonical rejection string.

---

### Q5: What happens when an answer is not in the uploaded documents?
**Answer**:
The system retrieves the closest chunks, but because their semantic content does not address the question, the LLM complies with the strict system instructions and replies:
`"I don't have enough information to answer that."`
It does not answer from general internet knowledge or pre-training memory.

---

### Q6: What is document chunking?
**Answer**:
Chunking is the process of breaking long, continuous documents into smaller, manageable text segments (nodes) suitable for embedding models and LLM context windows.

---

### Q7: Why do we overlap chunks?
**Answer**:
Chunk overlap (e.g., 64 characters/tokens between adjacent chunks) ensures that semantic meaning, grammatical clauses, and context that span across the splitting boundary are not severed or lost during segmentation.

---

### Q8: What are our chunking hyperparameters and how were they chosen?
**Answer**:
- `chunk_size = 512` characters/tokens.
- `chunk_overlap = 64` characters/tokens.
512 tokens is optimal for the `all-MiniLM-L6-v2` embedding model (which has a maximum sequence length of 256-512 word pieces) while keeping the context focused enough to retrieve precise paragraphs without diluting similarity scores.

---

### Q9: What are embeddings?
**Answer**:
Embeddings are dense mathematical vector representations of text in a continuous multi-dimensional space ($\mathbb{R}^{d}$). Semantically similar phrases are mapped closer together in this geometric space, allowing mathematical distance metrics (e.g., cosine similarity) to capture conceptual relationships rather than mere keyword overlap.

---

### Q10: Why did we select `sentence-transformers/all-MiniLM-L6-v2`?
**Answer**:
1. **Local & Free**: Runs 100% locally on CPU without paid API tokens or latency bottlenecks.
2. **High Efficiency**: 384 dimensions allow fast vector computation and compact ChromaDB storage.
3. **Benchmark Performance**: Proven high performance on the Massive Text Embedding Benchmark (MTEB) for semantic search and retrieval tasks.

---

### Q11: What is the embedding dimension of our model?
**Answer**:
384 dimensions.

---

### Q12: What is ChromaDB and why was it chosen?
**Answer**:
ChromaDB is an open-source, lightweight, embeddable vector database built for AI applications. We chose it because:
- It supports persistent on-disk storage (`PersistentClient`) directly in `./chroma_db`.
- It requires no complex external database server setup (like Milvus or Pinecone).
- Seamless native integration with LlamaIndex.
- Native metadata filtering and collection lifecycle management.

---

### Q13: What is similarity search and which metric is used?
**Answer**:
Similarity search is the process of finding vector representations in a database closest to a query vector. We use **Cosine Similarity**, which measures the cosine of the angle between two non-zero vectors:
$$\text{Cosine Similarity} = \frac{A \cdot B}{\|A\|_2 \|B\|_2}$$
Values range between 0 (orthogonal/unrelated) and 1 (identical direction/high semantic similarity).

---

### Q14: What is `similarity_top_k`?
**Answer**:
`top_k` specifies the number of highest-scoring chunks retrieved from ChromaDB to construct the context for the LLM. In our application, default `top_k = 3`, configurable from 1 to 8 in the sidebar.

---

### Q15: What is LlamaIndex and what role does it play?
**Answer**:
LlamaIndex is a data framework designed to connect custom data sources to Large Language Models. In our project, it manages:
- Document loading (`SimpleDirectoryReader`)
- Text chunking (`SentenceSplitter`)
- Embedding orchestration (`HuggingFaceEmbedding`)
- Vector store abstraction (`ChromaVectorStore`)
- Query engine and retriever abstractions

---

### Q16: Why use an open-source LLM like Llama 3.1?
**Answer**:
- **Transparency & Reproducibility**: Open weights allow academic inspection and independent evaluation.
- **Cost Efficiency**: Eliminates exorbitant per-token billing of proprietary closed models.
- **Data Sovereignty**: Can be hosted on private enterprise servers or locally via Ollama.

---

### Q17: Why use Groq as the default LLM provider?
**Answer**:
Groq utilizes custom hardware called Language Processing Units (LPUs). LPUs deliver deterministic, ultra-fast token generation speeds (~300-500 tokens/sec for Llama 3.1 8B), enabling instant, real-time interactive document QA.

---

### Q18: What is Llama 3.1 8B Instant?
**Answer**:
It is Meta's state-of-the-art 8-billion parameter instruction-tuned open-weights model, optimized for low-latency reasoning, tool-use, and precise context adherence.

---

### Q19: How is document metadata preserved throughout the pipeline?
**Answer**:
When `pypdf` extracts pages, page labels and file paths are saved to `Document.metadata`. During chunking, `SentenceSplitter` propagates this dictionary to every generated `TextNode`. When stored in ChromaDB, the metadata fields (`file_name`, `page_label`, `chunk_id`) are preserved alongside the embedding vectors.

---

### Q20: How are source citations extracted and displayed?
**Answer**:
During retrieval, the `retriever.py` module inspects the `NodeWithScore` objects, extracting the `file_name`, `page_label`, similarity score, and chunk preview. The Streamlit UI formats these into dedicated visual cards beneath each answer.

---

### Q21: What is the purpose of Retrieval Inspection / Debug Mode?
**Answer**:
The sidebar checkbox enables real-time visualization of raw retrieved chunks, similarity scores, and metadata before LLM synthesis. This proves to project evaluators that the RAG pipeline is actively searching vector space and not synthesizing from thin air.

---

### Q22: How does the system handle session state and chat history?
**Answer**:
Streamlit's `st.session_state` preserves conversation turns (`st.session_state.chat_history`), loaded vector indices, and configuration across UI reruns, with a dedicated "Clear Chat" action to reset state.

---

### Q23: Why avoid rebuilding the ChromaDB database on every rerun?
**Answer**:
Streamlit executes the script from top to bottom on each interaction. To avoid re-embedding hundreds of chunks on every button click, `load_vector_index()` connects to the existing ChromaDB directory on startup, reloading the persisted collection instantly.

---

### Q24: What happens if GROQ_API_KEY is not configured?
**Answer**:
The application gracefully prevents crashes. It presents a password-masked input in the sidebar for the user to provide their key, displays clear setup instructions, and allows retrieval testing even without the API key.

---

### Q25: How is Ollama integrated into the system?
**Answer**:
`src/llm.py` includes a health check function `check_ollama_available()` that queries `http://localhost:11434/api/tags`. If active, the user can toggle to Ollama in the UI for a 100% offline, private RAG pipeline.

---

### Q26: How is the system evaluated?
**Answer**:
The project includes a benchmark suite in `tests/evaluation_questions.json` containing 18 curated questions:
- **Direct Questions**: Single factual lookups.
- **Multi-Chunk Questions**: Cross-paragraph synthesis.
- **Outside-Knowledge Questions**: Irrelevant questions to verify hallucination rejection.
The script `tests/run_evaluation.py` executes these queries and computes authentic retrieval hit rates and rejection metrics.

---

### Q27: What is the difference between dense and sparse retrieval?
**Answer**:
- **Sparse Retrieval (e.g., BM25)**: Relies on exact term frequency and inverse document frequency matching; struggles with synonyms and phrasing variations.
- **Dense Retrieval (e.g., our ChromaDB + MiniLM)**: Relies on semantic latent space embeddings; successfully retrieves relevant answers even when no exact matching keywords are used.

---

### Q28: What is a Vector Store Index?
**Answer**:
A Vector Store Index is an in-memory or on-disk data structure that indexes vector embeddings using algorithms like HNSW (Hierarchical Navigable Small World) for fast approximate nearest neighbor (ANN) lookups over large document collections.

---

### Q29: What are the primary limitations of the current system?
**Answer**:
1. Scanned image-only PDFs require OCR before text extraction.
2. Tables with complex multi-column layouts can lose horizontal alignment when serialized to plain text.
3. Very large collections with millions of chunks would benefit from an approximate search cluster rather than local single-node ChromaDB.

---

### Q30: What future improvements could be added?
**Answer**:
1. **Hybrid Search**: Combining dense vector embeddings with BM25 keyword search using Reciprocal Rank Fusion (RRF).
2. **Re-ranking**: Adding a secondary cross-encoder (such as Cohere Rerank or BGE-Reranker) to re-score the top 20 retrieved candidates before sending the top 3 to the LLM.
3. **Query Expansion & HyDE**: Using Hypothetical Document Embeddings to enrich user questions before searching vector space.

---

### Q31: How do you explain the difference between Cosine Distance and Cosine Similarity?
**Answer**:
Cosine Similarity measures orientation alignment: $\cos(\theta) \in [-1, 1]$ (or $[0, 1]$ for normalized non-negative vectors). Cosine Distance is simply $1 - \text{Cosine Similarity}$. Higher similarity means closer relevance, whereas higher distance means greater dissimilarity.

---

### Q32: What is the role of Temperature in the LLM?
**Answer**:
Temperature controls the randomness of probability distribution during token sampling. A high temperature ($\ge 0.8$) produces creative, varied responses. A low temperature ($0.1$ in our project) flattens token probabilities to select the most probable, deterministic tokens, which is essential for factual, grounded RAG answering.
