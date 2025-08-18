# 🚀 HackRx 6.0 — Intelligent Query Retrieval

![HackRx 6.0 Banner](https://d8it4huxumps7.cloudfront.net/uploads/images/686eb5d3243dd_hackrx-60.jpg?d=980x520)

> **A Next-Gen RAG System for Insurance, Legal, HR, and Compliance Document Intelligence**

---

![Banner](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square) ![License](https://img.shields.io/github/license/KishoreMuruganantham/HackRx-6.0-Intelligent-Query-Retrieval?style=flat-square) ![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?style=flat-square) ![LangChain](https://img.shields.io/badge/LangChain-LLM-orange?style=flat-square)

Intelligent Query Retrieval is a high-performance, explainable Retrieval-Augmented Generation (RAG) backend for extracting answers from large and complex documents (PDFs, DOCX, emails). Optimized for speed, semantic accuracy, and professional deployment.

---

## 👥 Team Details

**Team Name:** Team Maverick

| Member                | Role          | GitHub                                           |
|-----------------------|--------------|--------------------------------------------------|
| Kishore Muruganantham | Team Leader  | [KishoreMuruganantham](https://github.com/KishoreMuruganantham) |
| Mugundhan Y           | Member       | [MugundhanY](https://github.com/MugundhanY)      |
| Mukundh A P           | Member       | [MukundhArul](https://github.com/MukundhArul)    |
| Praveen Kumar R       | Member       | [praveen647](https://github.com/praveen647)      |
| Prince Raj J          | Member       | [the-ai-developer](https://github.com/the-ai-developer) |

---

## 📋 Table of Contents

- [✨ Features](#features)
- [⚡ Quick Setup](#quick-setup)
- [🔗 API Usage](#api-usage)
- [🔄 Retraining & Regeneration](#retraining--regeneration)
- [📊 Dataset Insights](#dataset-insights)
- [📑 License](#license)

---

## ✨ Features

- **Multi-LLM Support**: Gemini, Groq, DeepSeek, OpenAI, and OpenRouter (via LangChain)
- **Automated Document Processing**: Ingests PDFs and creates semantic vectorstores
- **Robust Semantic Search**: FAISS-powered retrieval for clause matching and context-aware answers
- **Explainable Responses**: Outputs JSON with traceable, document-grounded answers
- **Lightning-Fast API**: Asynchronous, parallel processing for high throughput
- **Secure & Tokenized**: API endpoints protected with token-based authentication

---

## ⚡ Quick Setup

> **Requirements:** Python 3.9+, pip, API keys for supported LLMs

1. **Clone & Install**
   ```bash
   git clone https://github.com/KishoreMuruganantham/HackRx-6.0-Intelligent-Query-Retrieval.git
   cd HackRx-6.0-Intelligent-Query-Retrieval
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   - Add your API keys in a `.env` file or as environment variables:
     ```
     GOOGLE_API_KEY=your-google-api-key
     GROQ_API_KEY=your-groq-api-key
     DEEPSEEK_API_KEY=your-deepseek-api-key
     OPENAI_API_KEY=your-openai-api-key
     OPENROUTER_API_KEY=your-openrouter-api-key
     VALID_TOKEN=choose-a-strong-token
     ```

3. **Run the Backend**
   ```bash
   python main.py
   ```
   *Default port: 8000. Ngrok tunnel supported for remote access.*

---

## 🔗 API Usage

### 1️⃣ Health Check

**Endpoint:** `GET /health`

Returns status, cache info, and LLM pool size:
```json
{
  "status": "healthy",
  "cached_documents": 3,
  "llm_pool_size": 2
}
```

---

### 2️⃣ Document Ingestion

**Endpoint:** `POST /doc`

Add and cache a new document for retrieval (PDF URL as string):

**Request:**
```json
{
  "documents": "https://example.com/sample.pdf"
}
```

**Response:**
```json
{
  "message": "Document processing started",
  "doc_id": "abcdef123456"
}
```
If already cached:
```json
{
  "message": "Document already processed",
  "doc_id": "abcdef123456"
}
```

---

### 3️⃣ Query Retrieval

**Endpoint:** `POST /hackrx/run`  
**Auth:** Bearer token (`VALID_TOKEN`)

**Request:**
```json
{
  "documents": "https://example.com/sample.pdf",
  "questions": [
    "What is the claim eligibility?",
    "How to file a complaint?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "Eligibility for claims is outlined in Section 3...",
    "To file a complaint, follow procedure X..."
  ]
}
```

---

### 4️⃣ Cache Management

**Endpoint:** `DELETE /cache/{doc_hash}`

Clear vectorstore cache for a document:
```json
{
  "message": "Cache cleared for document: abcdef123456"
}
```

---

## 🔄 Retraining & Regeneration

- **Automatic:** Every new document triggers extraction, chunking, embedding, and vectorstore creation.
- **Manual:** Use `DELETE /cache/{doc_hash}` then re-upload to force regeneration.
- **Parallel Processing:** Async thread pools for document parsing and LLM querying.

**Pipeline:**
1. Download & extract text (PDF supported; see `get_pdf_text_cached`)
2. Smart chunking for optimal context (`get_text_chunks_optimized`)
3. Generate embeddings, build FAISS vectorstore (`create_vectorstore_optimized`)
4. Cache in memory + disk for speed & persistence

---

## 📊 Dataset Insights

- **Supported Formats:** PDF (fitz-based text extraction), with plans for DOCX & email
- **Smart Chunking:** Overlapping ~1500-character chunks retain context for semantic search
- **Semantic Search:** FAISS index enables rapid, high-accuracy retrieval
- **Explainability:** All answers traceable to document content; if not found, responds:  
  `"Information not available in the provided document"`

---

## 📑 License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for full text.

---

> _Built for HackRx 6.0 by Team Maverick — Professional, Scalable, and Ready for Real-World Deployment!_
