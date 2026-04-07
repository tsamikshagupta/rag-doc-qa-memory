# 🧠 RAG Document Q&A System with Long-term Memory

An intelligent, production-ready Document Q&A system built with **Google Gemini**, **ChromaDB**, **FastAPI**, and **Streamlit**. Uses Retrieval-Augmented Generation (RAG) with short-term + long-term memory and learning from user feedback.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      STREAMLIT UI                           │
│   Chat Interface │ Document Upload │ Admin Dashboard        │
└──────────────┬──────────────────────────────────────────────┘
               │ HTTP
┌──────────────▼──────────────────────────────────────────────┐
│                     FASTAPI SERVER                          │
│   /upload │ /ask │ /feedback │ /health │ /memory │ /metrics │
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│                    BACKEND SERVICES                         │
│                                                             │
│  ┌─────────────┐  ┌────────────┐  ┌──────────────────────┐ │
│  │  Document    │  │  Q&A       │  │  Retrieval Engine    │ │
│  │  Processor   │  │  Engine    │  │  (Query Expansion +  │ │
│  │  (Load/Chunk │  │  (RAG +    │  │   Re-ranking)        │ │
│  │   /Embed)    │  │   Cache)   │  │                      │ │
│  └──────┬──────┘  └─────┬──────┘  └──────────┬───────────┘ │
│         │               │                     │             │
│  ┌──────▼───────────────▼─────────────────────▼───────────┐ │
│  │              MEMORY SYSTEM                              │ │
│  │  SessionMemory (short-term) │ LongTermMemory (semantic) │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              ChromaDB (Vector Store)                    │ │
│  │  document_chunks │ interaction_history │ qa_cache │ ltm │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Gemini API (Wrapper)                       │ │
│  │  Embeddings (embedding-001) │ Generation (1.5-flash)   │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
RAG/
├── .env                         # GEMINI_API_KEY=your_key_here
├── requirements.txt             # All dependencies
├── rag_docqa_boilerplate.py     # Standalone monolithic version (all phases)
├── api.py                       # FastAPI server entry point
├── evaluate.py                  # SQuAD 2.0 evaluation runner
│
├── backend/
│   ├── __init__.py
│   ├── api/
│   │   └── __init__.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_processor.py   # Phase 1: Load, chunk, embed, ingest
│   │   └── qa_engine.py            # Phase 2: Full RAG answer cycle
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py              # Pydantic request/response models
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── session_memory.py       # Short-term conversation memory
│   │   └── long_term_memory.py     # Phase 3: Persistent memory + feedback
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py            # Query expansion + re-ranking
│   │   └── vector_db.py            # ChromaDB initialization
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # Centralized configuration
│       └── gemini_client.py        # Gemini API wrapper
│
├── frontend/
│   └── app.py                      # Streamlit chat UI + admin dashboard
│
└── data/
    ├── raw/                        # Place documents here for ingestion
    └── processed/                  # Processed outputs
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Your API Key

Edit `.env`:

```
GEMINI_API_KEY=your_actual_gemini_api_key
```

### 3. Start the FastAPI Server

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

### 4. Start the Streamlit Frontend

```bash
streamlit run frontend/app.py
```

Opens at: http://localhost:8501

### 5. Or Use the Standalone CLI

```bash
python rag_docqa_boilerplate.py
```

---

## 📡 API Endpoints

| Method | Endpoint    | Description                          |
|--------|-------------|--------------------------------------|
| POST   | `/upload`   | Upload and ingest a document         |
| POST   | `/ask`      | Ask a question (with session memory) |
| POST   | `/feedback` | Submit rating + optional correction  |
| GET    | `/health`   | Health check with collection stats   |
| GET    | `/memory`   | Long-term memory statistics          |
| GET    | `/metrics`  | System-wide metrics for dashboard    |

### Example: Ask a Question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main argument?", "session_id": "user1"}'
```

### Example: Upload a Document

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"
```

---

## 📊 Evaluation

Run the SQuAD 2.0 evaluation (100 examples):

```bash
python evaluate.py
```

Expected output:

```
=======================================================
  RAG Document Q&A — SQuAD 2.0 Evaluation Summary
=======================================================
  Metric                              Value
-------------------------------------------------------
  Total Examples                        100
  Answerable                             87
  Unanswerable                           13
-------------------------------------------------------
  Overall F1                          0.6500
  Exact Match Rate                    0.2800
  Answerable-only F1                  0.7100
  Unanswerable Accuracy               0.6150
=======================================================
```

---

## 🧠 Memory System

### Short-term Memory
- Keeps last 20 conversation exchanges per session
- Enables multi-turn context-aware queries
- Automatic query expansion using conversation history

### Long-term Memory
- Stores successful Q&A pairs with Gemini-classified topics
- Semantic retrieval via ChromaDB embeddings (filtered to rating >= 3)
- Tracks user corrections for learning from feedback
- Topic categories: legal, medical, finance, technical, general

### Query Cache
- Near-identical queries (cosine > 0.97) return cached answers
- TTL: 1 hour (configurable via `CACHE_TTL_HOURS`)

---

## ⚙️ Configuration

All settings are configurable via environment variables in `.env`:

| Variable                    | Default             | Description                     |
|-----------------------------|---------------------|---------------------------------|
| `GEMINI_API_KEY`            | (required)          | Google Gemini API key           |
| `CHUNK_SIZE`                | 512                 | Words per chunk                 |
| `CHUNK_OVERLAP`             | 64                  | Word overlap between chunks     |
| `TOP_K_RETRIEVAL`           | 5                   | Chunks retrieved per query      |
| `RELEVANCE_WEIGHT`          | 0.7                 | Weight for cosine similarity    |
| `RECENCY_WEIGHT`            | 0.3                 | Weight for recency scoring      |
| `SHORT_TERM_LIMIT`          | 20                  | Max session exchange pairs      |
| `MAX_CONTEXT_WORDS`         | 3000                | Max words in RAG prompt context |
| `CACHE_TTL_HOURS`           | 1.0                 | Cache expiration in hours       |
| `CACHE_SIMILARITY_THRESHOLD`| 0.97                | Min similarity for cache hit    |
| `GENERATION_MODEL`          | gemini-1.5-flash    | Gemini model for generation     |
| `EMBEDDING_MODEL`           | models/text-embedding-004| Gemini model for embeddings     |

---

## 🔒 Supported File Types

| Format   | Extension      | Parser               |
|----------|----------------|----------------------|
| PDF      | `.pdf`         | pypdf (PdfReader)    |
| Word     | `.docx`        | python-docx          |
| HTML     | `.html`, `.htm`| BeautifulSoup        |
| Text     | `.txt`         | Built-in             |
| Markdown | `.md`          | Built-in             |

---

## 🛡️ Error Handling

- All Gemini API calls wrapped with retry (60s wait on quota/rate errors)
- Graceful fallbacks: if LTM semantic search fails, falls back to recent pairs
- Cache lookup failures proceed with fresh generation
- API returns proper HTTP error codes with descriptive messages

---

## 📜 License

MIT
