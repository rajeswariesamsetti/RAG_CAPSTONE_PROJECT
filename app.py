"""
RAG Knowledge Assistant — Main Streamlit Application
B.Tech Final Year Capstone Project: Document Question Answering System

Pages: Landing → Sign Up → Login → Authenticated Dashboard
Dashboard sections: Overview, Documents, Ask AI, Settings
"""

import os
import sys
import html
import base64
from pathlib import Path

import streamlit as st

# ── Path Setup ──
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ── Page Config (MUST be first Streamlit call) ──
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Project Imports ──
from src.config import (
    DATA_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SIMILARITY_TOP_K, DEFAULT_GROQ_MODEL, DEFAULT_OLLAMA_MODEL,
    GROQ_API_KEY, SUPPORTED_EXTENSIONS, OUT_OF_CONTEXT_RESPONSE, EMBEDDING_DIM,
)
from utils.file_utils import (
    save_uploaded_file, get_uploaded_files, clear_data_directory,
    save_uploaded_file, get_uploaded_files, clear_data_directory, 
    is_supported_file, read_file_content, delete_single_file,
)
from src.document_loader import load_documents
from src.chunking import chunk_documents
from src.vector_store import (
    build_vector_index, load_vector_index, clear_vector_database,
    get_collection_stats, delete_document_vectors,
)
from src.rag_pipeline import RAGPipeline
from src.llm import check_ollama_available, fetch_available_groq_models
from src.auth import create_user, authenticate_user


# ══════════════════════════════════════════
# CSS LOADING
# ══════════════════════════════════════════
def load_css():
    css_path = BASE_DIR / "static" / "css" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

load_css()

HIDE_SIDEBAR = '<style>[data-testid="stSidebar"]{display:none!important}[data-testid="stSidebarCollapsedControl"]{display:none!important}</style>'

AUTH_PAGE_CSS = """
<style>
/* Center and constrain Streamlit page block on auth screens */
.main .block-container {
    max-width: 460px !important;
    padding-top: 3rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 1.25rem !important;
    padding-right: 1.25rem !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* Auth link container */
.auth-link-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    margin-top: 12px;
    margin-bottom: 8px;
    text-align: center;
}

.auth-link-prompt {
    color: #94A3B8;
    font-size: 14px;
    margin: 0;
    text-align: center;
}

/* Center navigation buttons wrapper below the auth card */
div[data-testid="stVerticalBlock"] > div.stButton,
div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"]:has(div.stButton),
div[data-testid="stElementContainer"] > div.stButton {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
    margin: 0 auto !important;
}

/* Compact, centered navigation buttons below auth card */
div[data-testid="stVerticalBlock"] > div.stButton > button,
div[data-testid="stElementContainer"] div.stButton > button {
    width: auto !important;
    min-width: 180px !important;
    max-width: 220px !important;
    padding: 8px 16px !important;
    font-size: 14px !important;
    line-height: 1.4 !important;
    height: auto !important;
    min-height: 38px !important;
    margin: 4px auto !important;
    display: inline-flex !important;
    justify-content: center !important;
    align-items: center !important;
    text-align: center !important;
    background: #111827 !important;
    border: 1px solid #263244 !important;
    color: #F8FAFC !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3) !important;
}

div[data-testid="stVerticalBlock"] > div.stButton > button:hover,
div[data-testid="stElementContainer"] div.stButton > button:hover {
    background: #1E2A3A !important;
    border-color: #6366F1 !important;
    color: #F8FAFC !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2) !important;
}

div[data-testid="stVerticalBlock"] > div.stButton > button:active,
div[data-testid="stElementContainer"] div.stButton > button:active {
    transform: translateY(0) !important;
    background: #151D2A !important;
}

/* Ensure the form submit button remains full width inside the auth card */
div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] > button {
    width: 100% !important;
    max-width: 100% !important;
}
</style>
"""

LANDING_CSS = """
<style>
.main .block-container {
    max-width: 1100px !important;
    padding: 2rem 1.5rem 3rem !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
</style>
"""

DASHBOARD_CSS = """
<style>
.main .block-container {
    max-width: 100% !important;
    padding: 1.5rem 2rem 2rem !important;
    margin-left: initial !important;
    margin-right: initial !important;
}
</style>
"""


# ══════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════
_DEFAULTS = {
    "page": "landing",
    "active_section": "dashboard",
    "authenticated": False,
    "user": None,
    "signup_success": False,
    "chat_history": [],
    "index": None,
    "index_loaded": False,
    "confirm_clear": False,
    "viewing_document": None,
    "llm_provider": "Groq",
    "groq_api_key": os.getenv("GROQ_API_KEY", GROQ_API_KEY),
    "selected_model": DEFAULT_GROQ_MODEL,
    "chunk_size": DEFAULT_CHUNK_SIZE,
    "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    "top_k": DEFAULT_SIMILARITY_TOP_K,
    "debug_mode": False,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════
def navigate(page: str):
    st.session_state.page = page

def set_section(section: str):
    st.session_state.active_section = section
    st.session_state.viewing_document = None

def _try_load_index():
    """Attempt to load existing ChromaDB index into session state."""
    if not st.session_state.index_loaded:
        try:
            idx = load_vector_index()
            if idx is not None:
                st.session_state.index = idx
                st.session_state.index_loaded = True
        except Exception:
            pass


# ══════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════
def render_landing():
    st.markdown(HIDE_SIDEBAR, unsafe_allow_html=True)
    st.markdown(LANDING_CSS, unsafe_allow_html=True)

    st.markdown("""
    <div class="landing-hero animate-in">
        <div class="hero-badge">🤖 AI-Powered Document Intelligence</div>
        <div class="hero-icon">📚</div>
        <h1 class="hero-title">RAG Knowledge Assistant</h1>
        <p class="hero-subtitle">
            Ask questions. Understand your documents.<br>
            Get accurate AI answers grounded in your knowledge base.
        </p>
    </div>
    """, unsafe_allow_html=True)

    _, c1, c2, c3, _ = st.columns([1.5, 1, 1, 1, 1.5])
    with c1:
        if st.button("🚀 Get Started", key="h_start", type="primary", use_container_width=True):
            navigate("signup"); st.rerun()
    with c2:
        if st.button("🔑 Login", key="h_login", use_container_width=True):
            navigate("login"); st.rerun()
    with c3:
        if st.button("📝 Sign Up", key="h_signup", use_container_width=True):
            navigate("signup"); st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Features
    st.markdown("""
    <div class="features-section">
        <h2 class="section-title">✨ Powerful Features</h2>
        <div class="features-grid">
            <div class="feature-card"><div class="feature-icon">📄</div>
                <h3>Document Upload</h3><p>Upload PDF and TXT files to build your AI knowledge base.</p></div>
            <div class="feature-card"><div class="feature-icon">🔍</div>
                <h3>Semantic Search</h3><p>Find information using state-of-the-art vector similarity search.</p></div>
            <div class="feature-card"><div class="feature-icon">🤖</div>
                <h3>AI-Powered Answers</h3><p>Intelligent responses from open-source LLMs via Groq.</p></div>
            <div class="feature-card"><div class="feature-icon">📚</div>
                <h3>Source Citations</h3><p>Every answer includes traceable source documents.</p></div>
            <div class="feature-card"><div class="feature-icon">🗄️</div>
                <h3>ChromaDB Storage</h3><p>Persistent vector storage for reliable retrieval.</p></div>
            <div class="feature-card"><div class="feature-icon">⚡</div>
                <h3>Fast Retrieval</h3><p>Lightning-fast search with 384-dim dense embeddings.</p></div>
            <div class="feature-card"><div class="feature-icon">🔐</div>
                <h3>Secure Auth</h3><p>Protected dashboard with session-based authentication.</p></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # How It Works
    st.markdown("""
    <div class="how-section">
        <h2 class="section-title">🔄 How It Works</h2>
        <div class="flow-container">
            <div class="flow-step"><div class="step-number">1</div>
                <div class="step-content"><h4>Upload Documents</h4><p>Add PDF or TXT files</p></div></div>
            <div class="flow-connector"></div>
            <div class="flow-step"><div class="step-number">2</div>
                <div class="step-content"><h4>Processing &amp; Chunking</h4><p>Text extracted &amp; split into chunks</p></div></div>
            <div class="flow-connector"></div>
            <div class="flow-step"><div class="step-number">3</div>
                <div class="step-content"><h4>Embeddings → ChromaDB</h4><p>Chunks converted to vectors &amp; stored</p></div></div>
            <div class="flow-connector"></div>
            <div class="flow-step"><div class="step-number">4</div>
                <div class="step-content"><h4>Semantic Retrieval</h4><p>Question finds most relevant chunks</p></div></div>
            <div class="flow-connector"></div>
            <div class="flow-step"><div class="step-number">5</div>
                <div class="step-content"><h4>LLM → Grounded Answer</h4><p>AI answers using only retrieved context</p></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="landing-footer">
        <p>📚 <strong>RAG Knowledge Assistant</strong> — B.Tech Final Year Capstone Project</p>
        <p>Built with Streamlit · LlamaIndex · ChromaDB · Groq · Sentence-Transformers</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# SIGNUP PAGE
# ══════════════════════════════════════════
def render_signup():
    st.markdown(HIDE_SIDEBAR, unsafe_allow_html=True)
    st.markdown(AUTH_PAGE_CSS, unsafe_allow_html=True)

    st.markdown("""
    <div class="auth-header">
        <div class="auth-logo">📚</div>
        <div class="auth-brand">RAG Knowledge Assistant</div>
        <h2 class="auth-title">Create Account</h2>
        <p class="auth-subtitle">Join RAG Knowledge Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("signup_form"):
        full_name = st.text_input("Full Name", placeholder="Enter your full name")
        email = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password", placeholder="Min 6 chars with a digit")
        confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
        submitted = st.form_submit_button("🚀 Create Account", use_container_width=True)
        if submitted:
            if password != confirm:
                st.error("⚠️ Passwords do not match.")
            else:
                ok, msg = create_user(full_name, email, password)
                if ok:
                    st.success(f"✅ {msg}")
                    st.session_state.signup_success = True
                else:
                    st.error(f"⚠️ {msg}")

    if st.session_state.signup_success:
        st.session_state.signup_success = False
        navigate("login")
        st.rerun()

    st.markdown("""
    <div class="auth-link-container">
        <p class="auth-link-prompt">Already have an account?</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Login →", key="to_login"):
        navigate("login")
        st.rerun()
    if st.button("← Back to Home", key="to_home_s"):
        navigate("landing")
        st.rerun()


# ══════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════
def render_login():
    st.markdown(HIDE_SIDEBAR, unsafe_allow_html=True)
    st.markdown(AUTH_PAGE_CSS, unsafe_allow_html=True)

    st.markdown("""
    <div class="auth-header">
        <div class="auth-logo">📚</div>
        <div class="auth-brand">RAG Knowledge Assistant</div>
        <h2 class="auth-title">Welcome Back</h2>
        <p class="auth-subtitle">Login to your account</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        email = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password", placeholder="Your password")
        submitted = st.form_submit_button("🔑 Login", use_container_width=True)
        if submitted:
            ok, user = authenticate_user(email, password)
            if ok:
                st.session_state.authenticated = True
                st.session_state.user = user
                st.session_state.page = "dashboard"
                st.session_state.active_section = "dashboard"
                st.rerun()
            else:
                st.error("⚠️ Invalid email or password. Please try again.")

    st.markdown("""
    <div class="auth-link-container">
        <p class="auth-link-prompt">Don't have an account?</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Sign Up →", key="to_signup"):
        navigate("signup")
        st.rerun()
    if st.button("← Back to Home", key="to_home_l"):
        navigate("landing")
        st.rerun()


# ══════════════════════════════════════════
# DASHBOARD (Authenticated)
# ══════════════════════════════════════════
def render_dashboard():
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
    _try_load_index()

    user = st.session_state.user or {}
    user_name = user.get("full_name", "User")
    user_email = user.get("email", "")
    initials = "".join(n[0].upper() for n in user_name.split()[:2]) if user_name else "U"

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:0.5rem 0 0.5rem;">
            <div style="font-size:1.5rem;">📚</div>
            <h3 style="margin:0.2rem 0 0;font-size:1rem;">RAG Assistant</h3>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-nav-title">NAVIGATION</p>', unsafe_allow_html=True)

        for label, key in [("🏠 Dashboard","dashboard"),("📁 Documents","documents"),
                           ("💬 Ask AI","ask_ai"),("⚙️ Settings","settings")]:
            btype = "primary" if st.session_state.active_section == key else "secondary"
            if st.button(label, key=f"nav_{key}", type=btype, use_container_width=True):
                set_section(key); st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sidebar-user">
            <div class="sidebar-user-avatar">{initials}</div>
            <div class="sidebar-user-info"><h4>{user_name}</h4><p>{user_email}</p></div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪 Logout", key="logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.session_state.page = "landing"
            st.rerun()

    # ── Content Router ──
    sec = st.session_state.active_section
    if sec == "dashboard":   _render_overview(user_name)
    elif sec == "documents": _render_documents()
    elif sec == "ask_ai":    _render_ask_ai()
    elif sec == "settings":  _render_settings()


# ──────────────────────────────────────────
# OVERVIEW
# ──────────────────────────────────────────
def _render_overview(user_name: str):
    stats = get_collection_stats()
    docs = get_uploaded_files()
    cc = stats.get("chunk_count", 0)

    st.markdown(f"""
    <div class="dash-header animate-in">
        <div class="dash-header-left"><h1>Welcome back, {html.escape(user_name)} 👋</h1>
        <p>Your AI Knowledge Assistant is ready</p></div>
        <div class="user-badge">📚 Knowledge Base</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metrics-grid animate-in">
        <div class="metric-card"><div class="metric-icon">🗄️</div>
            <div class="metric-value">{cc}</div><div class="metric-label">Vector Chunks</div></div>
        <div class="metric-card"><div class="metric-icon">📄</div>
            <div class="metric-value">{len(docs)}</div><div class="metric-label">Documents</div></div>
        <div class="metric-card"><div class="metric-icon">🧩</div>
            <div class="metric-value">{cc}</div><div class="metric-label">Indexed Chunks</div></div>
        <div class="metric-card"><div class="metric-icon">🧠</div>
            <div class="metric-value">{EMBEDDING_DIM}</div><div class="metric-label">Embed Dim</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🚀 Quick Actions")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📁 Documents", use_container_width=True, type="primary", key="qa_docs"):
            set_section("documents"); st.rerun()
    with c2:
        if st.button("💬 Ask AI", use_container_width=True, type="primary", key="qa_ai"):
            set_section("ask_ai"); st.rerun()
    with c3:
        if st.button("⚙️ Settings", use_container_width=True, key="qa_set"):
            set_section("settings"); st.rerun()

    st.markdown("---")
    if cc > 0:
        st.markdown(f'<span class="status-ready">● Vector DB: {cc} chunks ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-empty">● Vector DB: Empty — upload docs &amp; build index</span>', unsafe_allow_html=True)


# ──────────────────────────────────────────
# DOCUMENTS
# ──────────────────────────────────────────
def _render_documents():
    st.markdown("""
    <div class="dash-header animate-in">
        <div class="dash-header-left"><h1>📁 Document Management</h1>
        <p>Upload, preview, and manage your knowledge base documents</p></div>
    </div>
    """, unsafe_allow_html=True)

    # Document viewer takes over the page
    if st.session_state.viewing_document:
        _render_doc_viewer(st.session_state.viewing_document)
        return

    # Upload zone
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-icon">📤</div>
        <p class="upload-text">Upload your documents</p>
        <p class="upload-hint">PDF or TXT files supported</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose files", type=["pdf","txt"],
                                accept_multiple_files=True, key="doc_up",
                                label_visibility="collapsed")
    if uploaded:
        saved = sum(1 for u in uploaded if is_supported_file(u.name) and save_uploaded_file(u))
        if saved:
            st.success(f"✅ {saved} file(s) uploaded successfully!")

    docs = get_uploaded_files()

    # Build Index quick-button
    if docs:
        _, bc = st.columns([3, 1])
        with bc:
            if st.button("📦 Build Index", type="primary", use_container_width=True, key="build_doc"):
                _do_build_index()

    # Document cards
    st.markdown("### 📚 Uploaded Documents")
    if not docs:
        st.info("No documents uploaded yet. Use the area above to add files.")
        return

    for doc in docs:
        ext = doc.get("extension", "")
        icon = "📕" if ext == ".pdf" else "📄"
        ftype = "PDF Document" if ext == ".pdf" else "TXT Document"

        st.markdown(f"""
        <div class="doc-card">
            <div class="doc-card-header">
                <div class="doc-card-icon">{icon}</div>
                <div class="doc-card-info"><h4>{html.escape(doc['name'])}</h4>
                <p>{ftype} · {doc['size_kb']} KB</p></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        _, b1, b2 = st.columns([4, 1, 1])
        with b1:
            if st.button("📖 Open", key=f"o_{doc['name']}", use_container_width=True):
                st.session_state.viewing_document = doc["name"]; st.rerun()
        with b2:
            if st.button("🗑️", key=f"d_{doc['name']}", use_container_width=True):
                st.session_state[f"cdel_{doc['name']}"] = True; st.rerun()

        if st.session_state.get(f"cdel_{doc['name']}"):
            st.warning(f"Delete **{doc['name']}** and its vectors from ChromaDB?")
            y, n = st.columns(2)
            with y:
                if st.button("✅ Confirm", key=f"cy_{doc['name']}"):
                    delete_single_file(doc["name"])
                    try: delete_document_vectors(doc["name"])
                    except Exception: pass
                    try:
                        ni = load_vector_index()
                        st.session_state.index = ni
                        st.session_state.index_loaded = ni is not None
                    except Exception:
                        st.session_state.index = None; st.session_state.index_loaded = False
                    st.session_state[f"cdel_{doc['name']}"] = False
                    st.success(f"✅ Deleted {doc['name']}"); st.rerun()
            with n:
                if st.button("❌ Cancel", key=f"cn_{doc['name']}"):
                    st.session_state[f"cdel_{doc['name']}"] = False; st.rerun()


def _render_doc_viewer(filename: str):
    """Render document preview for a specific file."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        st.error(f"⚠️ File not found: {filename}")
        st.session_state.viewing_document = None
        return

    icon = "📕" if filepath.suffix.lower() == ".pdf" else "📄"
    size_kb = round(filepath.stat().st_size / 1024, 2)

    st.markdown(f"""
    <div class="doc-viewer animate-in">
        <div class="doc-viewer-header"><h3>{icon} {html.escape(filename)}</h3></div>
        <div class="doc-viewer-meta">
            <span>📏 {size_kb} KB</span>
            <span>📁 {filepath.suffix.upper().lstrip('.')} file</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("← Back to Documents", key="close_viewer"):
        st.session_state.viewing_document = None; st.rerun()

    st.markdown("---")

    if filepath.suffix.lower() == ".txt":
        content = read_file_content(filename)
        if content:
            st.markdown(f'<div class="doc-viewer-content">{html.escape(content)}</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ File appears to be empty or unreadable.")

    elif filepath.suffix.lower() == ".pdf":
        try:
            pdf_bytes = filepath.read_bytes()
            if len(pdf_bytes) < 5 * 1024 * 1024:
                b64 = base64.b64encode(pdf_bytes).decode()
                st.markdown(
                    f'<iframe src="data:application/pdf;base64,{b64}" '
                    f'width="100%" height="600" style="border:1px solid #263244;border-radius:0.75rem;"></iframe>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("📄 PDF too large for inline preview. Use download below.")
            st.download_button("📥 Download PDF", pdf_bytes, filename, "application/pdf", use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Unable to preview PDF: {e}")

    # Extracted text fallback
    with st.expander("📝 Extracted Text Content", expanded=False):
        text = read_file_content(filename)
        if text:
            st.text_area("", text, height=300, disabled=True, label_visibility="collapsed")
        else:
            st.caption("No extractable text content.")


# ──────────────────────────────────────────
# ASK AI (Chat)
# ──────────────────────────────────────────
def _render_ask_ai():
    st.markdown("""
    <div class="dash-header animate-in">
        <div class="dash-header-left"><h1>💬 Ask AI</h1>
        <p>Answers are strictly grounded in your uploaded documents</p></div>
    </div>
    """, unsafe_allow_html=True)

    stats = get_collection_stats()
    if stats["chunk_count"] > 0:
        st.markdown(
            f'<span class="status-ready">● {stats["chunk_count"]} chunks · '
            f'{stats["document_count"]} docs</span>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ No knowledge base. Go to **Documents** → upload files → **Build Index**.")

    c1, c2 = st.columns([4, 1])
    with c1:
        if st.button("🧹 Clear Chat", key="clr_chat"):
            st.session_state.chat_history = []; st.rerun()
    with c2:
        st.session_state.debug_mode = st.checkbox("🔍 Debug", st.session_state.debug_mode, key="dbg")

    st.markdown("<br>", unsafe_allow_html=True)

    # Chat history
    for mi, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])})", expanded=False):
                    for si, s in enumerate(msg["sources"]):
                        pg = f"Page {s['page']}" if s.get("page") else "Page N/A"
                        sc = f"Score: {s['score']:.4f}" if s.get("score") is not None else ""
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-header"><span>📄 <b>{html.escape(s['filename'])}</b></span>
                            <span class="source-meta">{sc} · {pg}</span></div>
                            <div class="source-snippet">{html.escape(s.get('text_preview',''))}</div>
                        </div>""", unsafe_allow_html=True)
                        if st.button(f"📖 View", key=f"hs_{mi}_{si}", use_container_width=True):
                            st.session_state.viewing_document = s["filename"]
                            set_section("documents"); st.rerun()
            if msg.get("debug_chunks"):
                with st.expander("🔍 Debug", expanded=False):
                    st.json(msg["debug_chunks"])

    # Chat input
    query = st.chat_input("Ask a question about your documents...")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        if st.session_state.index is None:
            warn = "⚠️ **No knowledge base loaded.** Go to **Documents** and build the index first."
            with st.chat_message("assistant"):
                st.warning(warn)
            st.session_state.chat_history.append({"role": "assistant", "content": warn, "sources": []})
        else:
            with st.chat_message("assistant"):
                with st.spinner("Retrieving context and generating answer..."):
                    try:
                        pipe = RAGPipeline(
                            index=st.session_state.index,
                            similarity_top_k=st.session_state.top_k,
                            provider=st.session_state.llm_provider,
                            api_key=st.session_state.groq_api_key,
                            model_name=(st.session_state.selected_model
                                        if st.session_state.llm_provider == "Groq"
                                        else DEFAULT_OLLAMA_MODEL),
                        )
                        res = pipe.answer_question(query)
                        answer = res["answer"]
                        sources = res.get("sources", [])
                        raw = res.get("raw_chunks", [])

                        st.markdown(answer)

                        # Sources
                        if sources and answer != OUT_OF_CONTEXT_RESPONSE:
                            with st.expander(f"📚 Sources ({len(sources)})", expanded=True):
                                for si, s in enumerate(sources):
                                    pg = f"Page {s['page']}" if s.get("page") else "Page N/A"
                                    sc = f"Score: {s['score']:.4f}" if s.get("score") is not None else ""
                                    st.markdown(f"""
                                    <div class="source-card">
                                        <div class="source-header"><span>📄 <b>{html.escape(s['filename'])}</b></span>
                                        <span class="source-meta">{sc} · {pg}</span></div>
                                        <div class="source-snippet">{html.escape(s.get('text_preview',''))}</div>
                                    </div>""", unsafe_allow_html=True)
                                    if st.button("📖 View Source", key=f"ns_{si}", use_container_width=True):
                                        st.session_state.viewing_document = s["filename"]
                                        set_section("documents"); st.rerun()
                        elif answer == OUT_OF_CONTEXT_RESPONSE:
                            st.info("ℹ️ This question could not be answered from the uploaded documents.")

                        # Debug
                        dbg = None
                        if st.session_state.debug_mode and raw:
                            dbg = [{"rank": i+1, "filename": c["filename"], "page": c["page"],
                                    "similarity_score": c["score"], "chunk_id": c["chunk_id"],
                                    "text": c["text"]} for i, c in enumerate(raw)]
                            with st.expander("🔍 Retrieval Debug", expanded=True):
                                st.json(dbg)

                        st.session_state.chat_history.append({
                            "role": "assistant", "content": answer,
                            "sources": sources if answer != OUT_OF_CONTEXT_RESPONSE else [],
                            "debug_chunks": dbg,
                        })
                    except Exception as e:
                        err = f"❌ {e}"
                        st.error(err)
                        st.session_state.chat_history.append({"role":"assistant","content":err,"sources":[]})


# ──────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────
def _render_settings():
    st.markdown("""
    <div class="dash-header animate-in">
        <div class="dash-header-left"><h1>⚙️ Settings</h1>
        <p>Configure LLM, pipeline parameters, and manage your database</p></div>
    </div>
    """, unsafe_allow_html=True)

    # LLM
    st.markdown("### 🤖 LLM Configuration")
    lc1, lc2 = st.columns(2)
    with lc1:
        prov = st.selectbox("LLM Provider", ["Groq","Ollama"],
                            index=0 if st.session_state.llm_provider=="Groq" else 1,
                            key="s_prov")
        st.session_state.llm_provider = prov

    if prov == "Groq":
        api_key = st.text_input("Groq API Key", value=st.session_state.groq_api_key,
                                type="password", help="https://console.groq.com/keys", key="s_key")
        st.session_state.groq_api_key = api_key
        if not api_key or not api_key.strip():
            st.warning("⚠️ No API key configured.")
        elif not api_key.startswith("gsk_"):
            st.warning("⚠️ Groq keys typically start with `gsk_`.")

        with lc2:
            models = fetch_available_groq_models(api_key)
            didx = models.index(st.session_state.selected_model) if st.session_state.selected_model in models else 0
            sel = st.selectbox("Model", models, index=didx, key="s_model")
            st.session_state.selected_model = sel
        st.caption(f"Active: `{sel}`")
    else:
        if check_ollama_available():
            st.success("✅ Ollama active")
        else:
            st.warning("⚠️ Ollama unreachable — run `ollama serve`")
        st.caption(f"Model: `{DEFAULT_OLLAMA_MODEL}`")

    st.markdown("---")

    # Pipeline
    st.markdown("### 📐 Pipeline Parameters")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.session_state.chunk_size = st.slider("Chunk Size", 128, 1024,
                                                st.session_state.chunk_size, 64, key="s_cs")
    with p2:
        st.session_state.chunk_overlap = st.slider("Overlap", 0, 256,
                                                   st.session_state.chunk_overlap, 16, key="s_co")
    with p3:
        st.session_state.top_k = st.slider("Top-K", 1, 8,
                                           st.session_state.top_k, key="s_tk")

    st.markdown("---")

    # DB Management
    st.markdown("### 🗄️ Database Management")
    d1, d2 = st.columns(2)
    with d1:
        if st.button("📦 Rebuild Index", type="primary", use_container_width=True, key="s_rebuild"):
            _do_build_index()
    with d2:
        if st.button("🗑️ Reset Database", use_container_width=True, key="s_reset"):
            st.session_state.confirm_clear = True

    if st.session_state.confirm_clear:
        st.warning("⚠️ This will delete ALL vectors and uploaded files.")
        y, n = st.columns(2)
        with y:
            if st.button("✅ Confirm Reset", key="s_cy"):
                clear_vector_database(); clear_data_directory()
                st.session_state.index = None; st.session_state.index_loaded = False
                st.session_state.chat_history = []; st.session_state.confirm_clear = False
                st.success("✅ Database and files cleared."); st.rerun()
        with n:
            if st.button("❌ Cancel", key="s_cn"):
                st.session_state.confirm_clear = False; st.rerun()

    st.markdown("---")

    # Status
    st.markdown("### 📊 Status & Metrics")
    stats = get_collection_stats()
    if stats["chunk_count"] > 0:
        st.markdown(f'<span class="status-ready">● {stats["chunk_count"]} chunks ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-empty">● Vector DB empty</span>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-card"><div class="metric-icon">📄</div>
            <div class="metric-value">{stats.get('document_count',0)}</div><div class="metric-label">Documents</div></div>
        <div class="metric-card"><div class="metric-icon">🧩</div>
            <div class="metric-value">{stats['chunk_count']}</div><div class="metric-label">Chunks</div></div>
        <div class="metric-card"><div class="metric-icon">🧠</div>
            <div class="metric-value">{EMBEDDING_DIM}</div><div class="metric-label">Embed Dim</div></div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────
# SHARED: Build Index
# ──────────────────────────────────────────
def _do_build_index():
    """Run the full build-index pipeline with progress feedback."""
    docs = get_uploaded_files()
    if not docs:
        st.error("No documents to index. Upload files first.")
        return
    with st.spinner("Building knowledge base..."):
        try:
            prog = st.progress(0); status = st.empty()
            def _up(m, v): status.text(m); prog.progress(v)

            _up("Loading documents...", 15)
            loaded = load_documents(DATA_DIR)
            if not loaded:
                st.error("No text could be extracted."); return

            _up("Creating chunks...", 40)
            nodes = chunk_documents(loaded,
                                    chunk_size=st.session_state.chunk_size,
                                    chunk_overlap=st.session_state.chunk_overlap)
            _up("Generating embeddings & indexing...", 70)
            idx = build_vector_index(nodes, progress_callback=_up)
            st.session_state.index = idx; st.session_state.index_loaded = True
            _up("Complete!", 100)
            st.success(f"✅ Indexed {len(loaded)} sections → {len(nodes)} chunks")
            st.rerun()
        except Exception as e:
            st.error(f"❌ {e}")


# ══════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════
_page = st.session_state.page

if _page == "landing":
    render_landing()
elif _page == "signup":
    render_signup()
elif _page == "login":
    render_login()
elif _page == "dashboard":
    if not st.session_state.authenticated:
        st.session_state.page = "login"; st.rerun()
    render_dashboard()
else:
    render_landing()
