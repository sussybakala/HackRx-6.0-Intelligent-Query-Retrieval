from fastapi import FastAPI, Request, HTTPException, status, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel
from typing import List, Dict, Optional
from langchain_community.chat_models import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
from langchain_deepseek import ChatDeepSeek
import re
import os
import fitz
import requests
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda, RunnableParallel
from pyngrok import ngrok
from langchain_groq import ChatGroq
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_core.utils.utils import secret_from_env
from pydantic import Field, SecretStr
import nest_asyncio
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import pickle
from functools import lru_cache
import time
from typing import Tuple
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
os.environ["GOOGLE_API_KEY"] = "<>"
os.environ["GROQ_API_KEY"] = "<>"
os.environ["DEEPSEEK_API_KEY"] = "<>"
os.environ["OPENAI_API_KEY"] = "<>"
os.environ["OPENROUTER_API_KEY"] = "<>"

# Global variables for caching
vectorstore_cache: Dict[str, FAISS] = {}
text_cache: Dict[str, str] = {}
embeddings_model = None
llm_pool = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global embeddings_model, llm_pool
    logger.info("Initializing models...")

    # Initialize embeddings model once
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create a pool of LLM instances for better concurrency
    for _ in range(3):  # Adjust based on your needs
        llm = ChatGoogleGenerativeAI(
            api_key="AIzaSyAKPDCUeTbalrTrFuSYh0Ae1g-fr0Szy88",
            model="gemini-2.0-flash",
            max_tokens=8000
        )
        llm_pool.append(llm)

    logger.info("Models initialized successfully")
    yield
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

VALID_TOKEN = "<>"
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != VALID_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token"
        )

class InputRequest(BaseModel):
    documents: str
    questions: List[str]

class Doc(BaseModel):
    documents: str

def generate_document_hash(url: str) -> str:
    """Generate a hash for document URL to use as cache key"""
    return hashlib.md5(url.encode()).hexdigest()

@lru_cache(maxsize=10)
def get_pdf_text_cached(pdf_url: str) -> str:
    """Cached version of PDF text extraction"""
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()

        pdf_data = response.content
        text = ""

        with fitz.open(stream=pdf_data, filetype="pdf") as pdf_document:
            # Use threading for page processing
            def extract_page_text(page_num):
                page = pdf_document[page_num]
                return page.get_text()

            with ThreadPoolExecutor(max_workers=4) as page_executor:
                page_texts = list(page_executor.map(
                    extract_page_text,
                    range(pdf_document.page_count)
                ))

            text = "".join(page_texts)

        return text
    except Exception as e:
        logger.error(f"Error extracting PDF from {pdf_url}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {e}")

def get_pdf_text_parallel(pdf_urls: List[str]) -> str:
    """Extract text from multiple PDFs in parallel"""
    if len(pdf_urls) == 1:
        return get_pdf_text_cached(pdf_urls[0])

    with ThreadPoolExecutor(max_workers=min(len(pdf_urls), 4)) as url_executor:
        texts = list(url_executor.map(get_pdf_text_cached, pdf_urls))

    return "".join(texts)

def get_text_chunks_optimized(text: str) -> List[str]:
    """Optimized text chunking with better parameters"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Smaller chunks for better retrieval
        chunk_overlap=200,  # Reduced overlap
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def create_vectorstore_optimized(text_chunks: List[str], doc_hash: str) -> FAISS:
    """Create vectorstore with optimized batch processing"""
    global embeddings_model

    if not text_chunks:
        raise ValueError("No text chunks provided")

    # Process chunks in batches for better memory management
    batch_size = 50
    vectorstore = None

    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]

        if vectorstore is None:
            vectorstore = FAISS.from_texts(batch, embeddings_model)
        else:
            batch_vectorstore = FAISS.from_texts(batch, embeddings_model)
            vectorstore.merge_from(batch_vectorstore)

    # Cache the vectorstore
    vectorstore_cache[doc_hash] = vectorstore

    # Save to disk for persistence
    vectorstore.save_local(f"{doc_hash}-vectorstore")

    return vectorstore

async def process_document_async(documents: str) -> str:
    """Async document processing with caching"""
    doc_hash = generate_document_hash(documents)

    # Check if vectorstore already exists in cache
    if doc_hash in vectorstore_cache:
        logger.info(f"Using cached vectorstore for document: {doc_hash}")
        return doc_hash

    # Check if vectorstore exists on disk
    try:
        vectorstore = FAISS.load_local(
            f"{doc_hash}-vectorstore",
            embeddings_model,
            allow_dangerous_deserialization=True
        )
        vectorstore_cache[doc_hash] = vectorstore
        logger.info(f"Loaded vectorstore from disk for document: {doc_hash}")
        return doc_hash
    except:
        logger.info(f"Creating new vectorstore for document: {doc_hash}")

    # Process document in thread pool
    loop = asyncio.get_event_loop()

    # Extract text
    text = await loop.run_in_executor(
        executor, get_pdf_text_parallel, [documents]
    )

    # Create chunks
    chunks = await loop.run_in_executor(
        executor, get_text_chunks_optimized, text
    )

    # Create vectorstore
    vectorstore = await loop.run_in_executor(
        executor, create_vectorstore_optimized, chunks, doc_hash
    )

    return doc_hash

def get_optimized_prompt() -> str:
    """Optimized prompt template"""
    return """You are an AI assistant specialized in answering questions about insurance and legal documents.

Based STRICTLY on the provided context, answer the user's question concisely and accurately.

Rules:
- Only use information from the provided context
- If information is not available in context, state "Information not available in the provided document"
- For eligibility questions, provide clear rules and procedures
- Be concise but complete
- Keep your response straight forward and clear

Context:
{context}

Question: {query}

Answer:"""

def get_available_llm():
    """Get an available LLM from the pool"""
    global llm_pool
    if llm_pool:
        return llm_pool.pop(0)
    # Fallback: create new instance if pool is empty
    return ChatGoogleGenerativeAI(
        api_key="<>",
        model="gemini-2.0-flash",
        max_tokens=8000
    )

def return_llm_to_pool(llm):
    """Return LLM to pool"""
    global llm_pool
    if len(llm_pool) < 5:  # Maximum pool size
        llm_pool.append(llm)

async def get_conversation_chain_optimized(vectorstore: FAISS, questions: List[str]) -> List[str]:
    """Optimized conversation chain with batch processing and smart retrieval"""

    # Retrieve documents for all questions at once with deduplication
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}  # Reduced from 8 for faster retrieval
    )

    # Collect unique relevant documents across all questions
    all_docs = []
    seen_contents = set()

    # Process retrievals in parallel
    async def retrieve_for_question(question: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            retriever.get_relevant_documents,
            question
        )

    retrieval_tasks = [retrieve_for_question(q) for q in questions]
    all_retrievals = await asyncio.gather(*retrieval_tasks)

    # Deduplicate documents
    for docs in all_retrievals:
        for doc in docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                all_docs.append(doc)

    # Combine context
    combined_context = "\n\n---\n\n".join([doc.page_content for doc in all_docs])

    # Prepare prompts
    prompt_template = get_optimized_prompt()
    prompts = [
        prompt_template.format(context=combined_context, query=question)
        for question in questions
    ]

    # Process questions in parallel with LLM pool
    async def process_single_question(prompt: str) -> str:
        llm = get_available_llm()
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                executor,
                lambda: llm.invoke(prompt)
            )
            return response.content
        finally:
            return_llm_to_pool(llm)

    # Execute all questions in parallel
    tasks = [process_single_question(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)

    return results

@app.get('/')
async def base():
    return {"message": "Enhanced RAG API - Ready for high performance!"}

@app.post("/doc")
async def process_doc(request: Doc, background_tasks: BackgroundTasks):
    """Process document endpoint with background processing"""
    doc_hash = generate_document_hash(request.documents)

    if doc_hash in vectorstore_cache:
        return {"message": "Document already processed", "doc_id": doc_hash}

    # Process in background for better user experience
    background_tasks.add_task(process_document_async, request.documents)

    return {"message": "Document processing started", "doc_id": doc_hash}

@app.post("/hackrx/run", dependencies=[Depends(verify_token)])
async def secure_data_optimized(request: InputRequest):
    """Optimized main endpoint with comprehensive performance improvements"""
    start_time = time.time()

    try:
        documents = request.documents
        questions = request.questions

        if not questions:
            raise HTTPException(status_code=400, detail="No questions provided")

        # Step 1: Process document (async with caching)
        logger.info(f"Processing document for {len(questions)} questions")
        doc_hash = await process_document_async(documents)

        # Step 2: Get vectorstore (should be cached by now)
        vectorstore = vectorstore_cache.get(doc_hash)
        if not vectorstore:
            # Fallback: load from disk
            vectorstore = FAISS.load_local(
                f"{doc_hash}-vectorstore",
                embeddings_model,
                allow_dangerous_deserialization=True
            )
            vectorstore_cache[doc_hash] = vectorstore

        # Step 3: Process all questions with optimized pipeline
        logger.info("Processing questions with optimized pipeline")
        results = await get_conversation_chain_optimized(vectorstore, questions)

        processing_time = time.time() - start_time
        logger.info(f"Total processing time: {processing_time:.2f} seconds")
        print(results)
        return {
            "answers": results
        }

    except Exception as e:
        logger.error(f"Error in secure_data_optimized: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "cached_documents": len(vectorstore_cache),
        "llm_pool_size": len(llm_pool)
    }

@app.delete("/cache/{doc_hash}")
async def clear_cache(doc_hash: str):
    """Clear specific document from cache"""
    if doc_hash in vectorstore_cache:
        del vectorstore_cache[doc_hash]
        return {"message": f"Cache cleared for document: {doc_hash}"}
    return {"message": "Document not found in cache"}

def start_server():
    """Start the server with ngrok tunnel"""
    nest_asyncio.apply()
    ngrok.set_auth_token("<>")

    try:
        ngrok_tunnel = ngrok.connect(8000)
        print(f'Public URL: {ngrok_tunnel.public_url}')
    except Exception as e:
        print(f"Ngrok connection failed: {e}")
        print("Starting server without ngrok...")

    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=1  # Keep as 1 for development, increase for production
    )
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    start_server()
