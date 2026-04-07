# Offline RAG Document Q&A System

A completely local, privacy-first Retrieval-Augmented Generation (RAG) application built to chat with your documents without sending a single byte of data to external APIs. 

Built using **FastAPI**, **Streamlit**, **ChromaDB**, and **HuggingFace** models (`FLAN-T5` and `SentenceTransformers`).

---

## 🌟 Why this project?
Most modern Document Q&A systems rely on expensive, closed-source APIs like OpenAI or Google Gemini. This means sensitive documents (like healthcare records, legal contracts, or internal IP) are sent off-premise.

This project solves that by running a full RAG cycle—from semantic chunking and embedding to generative text responses—entirely on your local hardware. 

## ✨ Features

- **100% Offline Generation**: Uses Google's `flan-t5-base` model to answer your questions deterministically without external API calls.
- **Local Embeddings**: Leverages `all-MiniLM-L6-v2` (`sentence-transformers`) for fast, local vectorization of your documents.
- **Adaptive Memory**: 
  - *Short-Term*: Remembers the last 20 messages in your active session.
  - *Long-Term*: Logs your thumbs up/down feedback and maps interaction topics to improve future interactions.
- **Smart Retrieval**: Expands ambiguous questions using conversation history, and re-ranks vector results based on both semantic relevance and document recency.
- **Multi-Format Ingestion**: Drag and drop `.pdf`, `.docx`, `.txt`, `.md`, or `.html` files directly into the UI.
- **Admin Dashboard**: Real-time observability dashboard to monitor your vector footprint, cached queries, and LTM (Long-Term Memory) stats.

---

## 🛠️ Tech Stack
- **Backend Framework**: FastAPI (Async, REST endpoints)
- **Frontend App**: Streamlit
- **Generative AI**: `transformers` / FLAN-T5
- **Vector Database**: ChromaDB (Persistent local store)
- **Evaluation**: Custom SQuAD 2.0 evaluation runner (`evaluate.py`) for offline accuracy testing.

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies. (A virtual environment is highly recommended).

```bash
git clone https://github.com/tsamikshagupta/rag-doc-qa-memory.git
cd rag-doc-qa-memory
pip install -r requirements.txt
```

### 2. Start the Backend API
The FastAPI server acts as the brain. Run it in your first terminal window:

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```
> **Note:** The first time you launch this, PyTorch will automatically download the `FLAN-T5` and `MiniLM` model weights (~1-2 GB total) to your local cache.

### 3. Start the UI
In a separate terminal, launch the Streamlit frontend:

```bash
python -m streamlit run frontend/app.py
```
This will open up `http://localhost:8501` in your browser. From there, just upload a document from the sidebar and start asking questions!

---

## 🏗️ Project Structure
- `api.py` - Core execution point for the FastAPI server.
- `backend/services/` - Houses the document ingestion pipeline and Q&A engine.
- `backend/retrieval/` - Query expansion and ChromaDB integrations.
- `backend/llm/` - FLAN-T5 generation handler.
- `backend/embeddings/` - MiniLM SentenceTransformer handler.
- `backend/memory/` - Short and long-term conversation storage architectures.
- `frontend/app.py` - The Streamlit chat UI.

---

## 📝 Configuration
You can tweak the generation parameters and embedding models inside `backend/utils/config.py`.
By default:
- `MAX_CONTEXT_WORDS` is set to `300` to respect FLAN-T5's strict 512-token limit.
- `TOP_K_RETRIEVAL` fetches the top 5 most relevant chunks.

## 🤝 Contributing
Feel free to open an issue or submit a pull request if you want to expand the supported file types, upgrade the default LLM (like pointing to `Llama-3` or `Mistral` via `Ollama`), or improve the re-ranking formula!

---
*Built with ❤️ for privacy-first AI.*
