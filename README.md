# 📚 RAG Knowledge Assistant — Document Question Answering System

> **B.Tech Final Year Capstone Project**  
> An explainable, production-ready Retrieval-Augmented Generation (RAG) system with strict hallucination control, local embeddings, persistent vector storage, open-source LLM inference, and verifiable source citations.

---

## 1. Project Title
**RAG Knowledge Assistant — Document Question Answering System**

---

## 2. Problem Statement
Pre-trained Large Language Models (LLMs) are frozen in time and have no visibility into private, proprietary, or organizational documents (such as university handbooks, academic regulations, institutional bylaws, or internal manuals). When asked domain-specific questions, generic LLMs tend to generate **hallucinations**—convincing, fluent answers that are factually incorrect. Furthermore, feeding entire document corpora into LLM prompts exceeds token budget limits, is computationally inefficient, and provides no verifiable source citations.

---

## 3. Objectives
1. **End-to-End RAG Implementation**: Build a functional pipeline that extracts text from PDF and TXT documents, chunks with metadata preservation, embeds locally, and indexes into a persistent ChromaDB vector database.
2. **Elimination of Hallucinations**: Enforce strict system prompts and validation filters to ensure the system answers **only** from the uploaded documents. If the requested information is absent, it responds:
   > *"I don't have enough information to answer that."*
3. **Local & Cost-Free Embeddings**: Utilize `sentence-transformers/all-MiniLM-L6-v2` locally (384 dimensions) with zero external API fees.
4. **Fast Open-Source LLM Inference**: Integrate Meta's **Llama 3.1 8B Instant** served through Groq's high-speed Language Processing Unit (LPU) API, with local Ollama support.
5. **Transparent Source Citations**: Display exact source document name, page number, and cosine similarity score for every answer.
6. **Capstone Viva & Evaluation Readiness**: Provide comprehensive unit tests, an 18-question benchmark evaluation script, architecture documentation, report outline, and 30+ viva defense questions.

---

## 4. Key Features
- 🌐 **Modern Landing Page**: High-conversion dark-themed landing page with hero banner, live feature showcase, and step-by-step pipeline flow.
- 🔐 **Secure User Authentication**: Built-in SQLite authentication with PBKDF2-HMAC-SHA256 password hashing (100,000 iterations), salt generation, and session management.
- 📊 **Protected Multi-Section Dashboard**: Clean sidebar navigation across Overview (metric cards), Documents, Ask AI, and Settings.
- 👁️ **Inline Document Viewer**: Interactive document viewer with TXT preview, PDF embedded viewer, raw text inspection, and download capability.
- 🗑️ **Per-File Deletion**: Delete individual uploaded documents with automatic synchronization and targeted ChromaDB vector deletion.
- 📄 **Multi-Document Ingestion**: Support for multiple PDF and TXT documents simultaneously.
- ✂️ **Configurable Semantic Chunking**: Split by sentences (`SentenceSplitter`) with adjustable `chunk_size` and `chunk_overlap`.
- 🗄️ **Persistent ChromaDB**: Disk-persisted vector store (`./chroma_db`) that reloads instantly without re-embedding on every session.
- 🎯 **Semantic Similarity Retrieval**: Cosine similarity top-k search with adjustable top-k slider (1–8).
- 🛡️ **Zero-Tolerance Hallucination Control**: Model strictly prevented from using outside knowledge.
- 📑 **Visual Source Cards**: Expandable citation cards with filename, page numbers, and preview snippets with clickable "View Source" actions.
- 🔍 **Retrieval Inspection (Debug Mode)**: Real-time inspection of raw retrieved chunks and similarity scores before LLM generation.
- 💬 **Interactive Session Chat**: Full chat history with clear-chat functionality.
- 🧪 **Automated Benchmark Suite**: 18 pre-configured test queries evaluating direct, multi-chunk, and outside-knowledge questions.

---

## 5. System Architecture
```
              ┌─────────────────┐
              │ PDF / TXT Files │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │ Document Loader │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │    Chunking     │
              │ 512 / overlap64 │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │   Embeddings    │
              │ MiniLM-L6-v2    │
              │  (384 dense)    │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │    ChromaDB     │
              │  Vector Store   │
              └────────┬────────┘
                       ↓
User Question → Embedding → Similarity Retrieval (Top-K)
                       ↓
              ┌─────────────────┐
              │ Retrieved Chunks│
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │  Llama 3.1 8B   │
              │    via Groq     │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │ Grounded Answer │
              │   + Sources     │
              └─────────────────┘
```

---

## 6. Technology Stack
| Layer | Technology | Details |
|---|---|---|
| **Programming Language** | Python 3.10+ (tested on Python 3.12) | Core development |
| **Web Framework** | Streamlit (>= 1.35.0) | Interactive user interface |
| **RAG Framework** | LlamaIndex (>= 0.11.0) | Ingestion, chunking, retrieval orchestration |
| **Vector Database** | ChromaDB (>= 0.5.0) | Persistent vector storage (`./chroma_db`) |
| **Embedding Model** | Hugging Face Sentence Transformers | `all-MiniLM-L6-v2` (384 dimensions, runs locally) |
| **LLM Provider** | Groq API / Ollama | `llama-3.1-8b-instant` (default) / `llama3.1:8b` (local) |
| **PDF Extraction** | PyPDF (>= 4.0.0) | Extracting text and page metadata |
| **Environment Mgmt** | python-dotenv | Secure API key handling |
| **Testing & Evaluation** | pytest, custom benchmark runner | Factual metric reporting |

---

## 7. Project Structure
```
rag-knowledge-assistant/
│
├── app.py                     # Streamlit frontend & interactive dashboard
├── requirements.txt           # Verified pinned dependencies
├── README.md                  # Complete project documentation
├── .env.example               # Template for environment variables
├── .gitignore                 # Excludes venv, pycache, .env, chroma_db
│
├── data/                      # Ingested documents directory
│   ├── .gitkeep
│   └── sample_documents/      # Ready-to-use sample academic policy documents
│       ├── college_handbook.txt
│       └── academic_rules.txt
│
├── chroma_db/                 # Persistent ChromaDB vector database
│   └── .gitkeep
│
├── src/                       # Core RAG pipeline modules
│   ├── __init__.py
│   ├── config.py              # Central configuration & hyperparameters
│   ├── document_loader.py     # PDF/TXT ingestion with page metadata
│   ├── chunking.py            # SentenceSplitter chunking logic
│   ├── embeddings.py          # HuggingFace all-MiniLM-L6-v2 model loader
│   ├── vector_store.py        # ChromaDB index lifecycle & persistence
│   ├── retriever.py           # Similarity retrieval & chunk inspector
│   ├── llm.py                 # Groq / Ollama LLM provider factory
│   ├── rag_pipeline.py        # End-to-end question answering pipeline
│   └── prompts.py             # Strict RAG system & query prompts
│
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── file_utils.py          # File upload, validation, cleanup helpers
│   └── logging_utils.py       # Formatted console logger
│
├── tests/                     # Unit tests & benchmark evaluation
│   ├── test_chunking.py       # Chunk splitting & metadata tests
│   ├── test_retrieval.py      # Vector similarity retrieval tests
│   ├── test_rag.py            # Grounded QA & hallucination rejection tests
│   ├── evaluation_questions.json # 18 benchmark evaluation questions
│   └── run_evaluation.py      # Automated benchmark metrics calculator
│
└── docs/                      # Final year project documentation
    ├── architecture.md        # Technical architecture specifications
    ├── report_outline.md      # 14-chapter academic report outline
    └── viva_questions.md      # 32 viva examination questions & answers
```

---

## 8. Installation & Setup

### Prerequisites
- Python 3.10 or higher (Python 3.12 recommended)
- Git (optional)
- Free Groq Cloud account for API key (https://console.groq.com/keys)

### Step 1: Create and Activate Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 9. Environment Configuration
Create a `.env` file in the project root by copying `.env.example`:

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**macOS / Linux:**
```bash
cp .env.example .env
```

Edit `.env` and insert your Groq API key:
```ini
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
OLLAMA_BASE_URL=http://localhost:11434
```

> **Security Note**: Never commit your `.env` file to version control. It is already included in `.gitignore`.

---

## 10. Running the Application
Launch the Streamlit web application:
```bash
streamlit run app.py
```
The browser will automatically open to `http://localhost:8501`.

---

## 11. Step-by-Step Usage Guide

### 1. Upload Documents
- Expand **Upload Documents** in the sidebar.
- Upload one or more `.pdf` or `.txt` files (e.g. from `data/sample_documents/`).
- The application validates file formats and saves them safely to `./data/`.

### 2. Configure Parameters (Optional)
- Adjust **Chunk Size** (default: 512 characters).
- Adjust **Chunk Overlap** (default: 64 characters).
- Adjust **Top-K Retrieved Chunks** (default: 3).

### 3. Build Knowledge Base
- Click **"🔨 Build Index"** in the sidebar.
- Watch real-time progress indicators:
  - *"Loading documents..."*
  - *"Creating chunks..."*
  - *"Generating embeddings & saving to ChromaDB..."*
  - *"Knowledge base ready!"*
- The status badge will switch to **Vector DB: Ready** with exact chunk counts.

### 4. Ask Questions
- Enter questions in the chat bar at the bottom:
  - *Example 1*: `"What is the attendance requirement?"`
  - *Example 2*: `"What happens if a student violates the examination policy?"`
  - *Example 3*: `"Who won the FIFA World Cup?"` (Observe rejection!)

### 5. Inspect Citations & Debug Mode
- Expand the **📑 Sources Cited** container below any answer to view source files, page numbers, similarity scores, and text excerpts.
- Enable **🔍 Retrieval Inspection Mode** in the sidebar to see exact raw JSON chunks and similarity scores before LLM generation.

---

## 12. Running Automated Tests & Benchmark Evaluation

### Run Unit Tests
```bash
pytest tests/
```

### Run Benchmark Evaluation
To evaluate retrieval accuracy and hallucination rejection across all 18 benchmark questions:
```bash
python tests/run_evaluation.py
```
This produces an authentic report table:
```
===========================================================================
📊 EVALUATION METRICS REPORT
===========================================================================
Total Evaluated Questions:             18
  • Direct Questions:                  7
  • Multi-Chunk Questions:             5
  • Outside-Knowledge Questions:       6
---------------------------------------------------------------------------
In-Domain Target Document Hits:        12 / 12 (100.0%)
Retrieval Cosine Similarities:         Min=0.6124, Max=0.8871, Avg=0.7452
Outside-Knowledge Correct Rejections: 6 / 6 (100.0%)
===========================================================================
```
Results are saved to `tests/evaluation_results.json`.

---

## 13. Troubleshooting
| Issue | Cause | Solution |
|---|---|---|
| `GROQ_API_KEY is not configured` | Missing API key | Add `GROQ_API_KEY=...` to `.env` or paste in sidebar input. |
| `Ollama server is not reachable` | Ollama daemon not running | Run `ollama serve` in a terminal or switch provider to `Groq`. |
| `No knowledge base loaded` | Query submitted before indexing | Upload documents and click **"🔨 Build Index"** first. |
| `ChromaDB lock or database error` | Corrupt persistent state | Click **"🗑️ Reset DB"** in the sidebar to reset cleanly. |

---

## 14. Academic Deliverables & Viva Materials
- 📐 **Architecture Documentation**: [`docs/architecture.md`](docs/architecture.md)
- 📝 **14-Chapter Project Report Outline**: [`docs/report_outline.md`](docs/report_outline.md)
- 🎯 **32 Viva Examination Q&As**: [`docs/viva_questions.md`](docs/viva_questions.md)
