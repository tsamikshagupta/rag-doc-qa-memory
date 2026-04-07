"""
Streamlit Frontend for RAG Document Q&A System.

Features:
  - Chat interface with conversation history
  - Document upload UI
  - Feedback buttons (thumbs up/down + correction)
  - Admin dashboard with metrics
"""

import streamlit as st
import requests
import time
from datetime import datetime

# ─── Configuration ───
API_BASE = "http://localhost:8000"

# ─── Page Config ───
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .main-header {
        font-size: 2.2em;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2em;
    }
    .sub-header {
        color: #888;
        font-size: 1.1em;
        margin-bottom: 1.5em;
    }
    .source-chip {
        display: inline-block;
        background: #2d2d2d;
        border: 1px solid #444;
        border-radius: 12px;
        padding: 4px 12px;
        margin: 2px 4px;
        font-size: 0.8em;
        color: #ccc;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 1.2em;
        text-align: center;
    }
    .metric-value {
        font-size: 2em;
        font-weight: 700;
        color: #667eea;
    }
    .metric-label {
        color: #999;
        font-size: 0.85em;
        margin-top: 0.3em;
    }
    .stChatMessage {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State ───
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""


# ─── Helper Functions ───

def api_health() -> dict:
    """Check API health."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json() if r.status_code == 200 else {"status": "error"}
    except Exception:
        return {"status": "offline"}


def api_upload(file) -> dict:
    """Upload a document."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type or "application/octet-stream")}
        r = requests.post(f"{API_BASE}/upload", files=files, timeout=120)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_ask(query: str, session_id: str) -> dict:
    """Ask a question."""
    try:
        r = requests.post(
            f"{API_BASE}/ask",
            json={"query": query, "session_id": session_id},
            timeout=30,
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_feedback(query: str, answer: str, rating: int, correction: str = None) -> dict:
    """Submit feedback."""
    try:
        payload = {"query": query, "answer": answer, "rating": rating}
        if correction:
            payload["correction"] = correction
        r = requests.post(f"{API_BASE}/feedback", json=payload, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_metrics() -> dict:
    """Get system metrics."""
    try:
        r = requests.get(f"{API_BASE}/metrics", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def api_memory() -> dict:
    """Get memory stats."""
    try:
        r = requests.get(f"{API_BASE}/memory", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


# ─── Sidebar ───
with st.sidebar:
    st.markdown("### 📄 Document Upload")

    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT, HTML, MD)",
        type=["pdf", "docx", "txt", "html", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uf in uploaded_files:
            with st.spinner(f"Ingesting {uf.name}..."):
                result = api_upload(uf)
                if "error" in result:
                    st.error(f"❌ {uf.name}: {result['error']}")
                elif "detail" in result:
                    st.error(f"❌ {uf.name}: {result['detail']}")
                else:
                    st.success(f"✅ {uf.name} → doc_id: {result.get('doc_id', 'N/A')}")

    st.divider()

    # Health check
    st.markdown("### 🏥 System Status")
    health = api_health()
    if health.get("status") == "ok":
        st.success("● API Online")
        cols = health.get("collections", {})
        st.caption(f"Chunks: {cols.get('chunks', 0)} | LTM: {cols.get('ltm', 0)} | Cache: {cols.get('cache', 0)}")
    elif health.get("status") == "offline":
        st.error("● API Offline — start with: `uvicorn api:app --reload`")
    else:
        st.warning(f"● {health.get('status', 'unknown')}")

    st.divider()

    # Session controls
    st.markdown("### 💬 Session")
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = f"session_{int(time.time())}"
        st.session_state.last_query = ""
        st.session_state.last_answer = ""
        st.rerun()

    st.caption(f"ID: `{st.session_state.session_id}`")


# ─── Main Content ───
tab_chat, tab_admin = st.tabs(["💬 Chat", "📊 Admin Dashboard"])


# ─── Chat Tab ───
with tab_chat:
    st.markdown('<p class="main-header">🧠 RAG Document Q&A</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your uploaded documents</p>', unsafe_allow_html=True)

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📎 Sources", expanded=False):
                    for s in msg["sources"]:
                        src_name = s.get("source", "unknown").split("/")[-1].split("\\")[-1]
                        st.markdown(
                            f'<span class="source-chip">📄 {src_name} '
                            f'(score: {s.get("score", 0):.3f})</span>',
                            unsafe_allow_html=True,
                        )

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = api_ask(prompt, st.session_state.session_id)

            if "error" in result:
                answer = f"⚠️ Error: {result['error']}"
                sources = []
            elif "detail" in result:
                answer = f"⚠️ {result['detail']}"
                sources = []
            else:
                answer = result.get("answer", "No answer generated.")
                sources = result.get("sources", [])

            st.markdown(answer)
            if sources:
                with st.expander("📎 Sources", expanded=False):
                    for s in sources:
                        src_name = s.get("source", "unknown").split("/")[-1].split("\\")[-1]
                        st.markdown(
                            f'<span class="source-chip">📄 {src_name} '
                            f'(score: {s.get("score", 0):.3f})</span>',
                            unsafe_allow_html=True,
                        )

        st.session_state.messages.append({
            "role": "assistant", "content": answer, "sources": sources,
        })
        st.session_state.last_query = prompt
        st.session_state.last_answer = answer

    # Feedback section
    if st.session_state.last_answer:
        st.divider()
        st.markdown("**Rate the last answer:**")
        fc1, fc2, fc3 = st.columns([1, 1, 3])
        with fc1:
            if st.button("👍 Good", use_container_width=True):
                api_feedback(st.session_state.last_query, st.session_state.last_answer, 5)
                st.toast("✅ Positive feedback recorded!", icon="👍")
        with fc2:
            if st.button("👎 Bad", use_container_width=True):
                st.session_state["show_correction"] = True
        with fc3:
            if st.session_state.get("show_correction"):
                correction = st.text_input("What's a better answer?", key="correction_input")
                if st.button("Submit Correction"):
                    api_feedback(
                        st.session_state.last_query,
                        st.session_state.last_answer,
                        1,
                        correction=correction if correction else None,
                    )
                    st.toast("✅ Feedback with correction recorded!", icon="📝")
                    st.session_state["show_correction"] = False


# ─── Admin Dashboard Tab ───
with tab_admin:
    st.markdown("## 📊 Admin Dashboard")

    metrics = api_metrics()
    memory = api_memory()

    if not metrics:
        st.warning("Could not fetch metrics. Is the API running?")
    else:
        # Metric cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('total_documents', 0)}</div>
                <div class="metric-label">Document Chunks</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('total_interactions', 0)}</div>
                <div class="metric-label">Total Interactions</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics.get('cached_queries', 0)}</div>
                <div class="metric-label">Cached Queries</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            mem_stats = metrics.get("memory_stats", {})
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{mem_stats.get('total_qa_pairs', 0)}</div>
                <div class="metric-label">Q&A Pairs in Memory</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # Memory details
        if memory:
            st.markdown("### 🧠 Memory Insights")
            mc1, mc2 = st.columns(2)
            with mc1:
                st.metric("Total Q&A Pairs", memory.get("total_qa_pairs", 0))
                st.metric("Total Corrections", memory.get("total_corrections", 0))
                st.metric("LTM Embeddings", memory.get("ltm_embeddings", 0))
            with mc2:
                topics = memory.get("topics", {})
                if topics:
                    st.markdown("**Topic Distribution:**")
                    for topic, count in topics.items():
                        if count > 0:
                            st.progress(
                                min(count / max(sum(topics.values()), 1), 1.0),
                                text=f"{topic.capitalize()}: {count}",
                            )

        st.divider()

        # Refresh
        if st.button("🔄 Refresh Metrics", use_container_width=True):
            st.rerun()
