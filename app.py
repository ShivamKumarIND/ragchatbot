"""
Main Application - FastAPI Server
Provides REST API for document upload and chat functionality
"""
import os
import shutil
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from document_processor import DocumentProcessor
from vector_store import get_vector_store
from rag_chain import get_chatbot
from llm_loader import get_llm_loader

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Document-based Q&A system using RAG and LangChain",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize components
document_processor = DocumentProcessor()


# Pydantic models
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict] = []


class StatusResponse(BaseModel):
    status: str
    message: str
    details: Optional[dict] = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    print("=" * 50)
    print("Starting RAG Chatbot API")
    print("=" * 50)
    
    try:
        # Initialize LLM loader
        llm_loader = get_llm_loader()
        print(f"Available LLMs: {llm_loader.list_available_llms()}")
        
        # Initialize vector store
        vector_store = get_vector_store()
        doc_count = vector_store.get_collection_count()
        print(f"Vector store loaded: {doc_count} documents")
        
        # Initialize chatbot
        chatbot = get_chatbot()
        print("Chatbot initialized")
        
        print("=" * 50)
        print("✓ Server ready!")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during startup: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload",
            "chat": "/chat",
            "history": "/history",
            "clear": "/clear",
            "status": "/status"
        }
    }


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status"""
    try:
        vector_store = get_vector_store()
        doc_count = vector_store.get_collection_count()
        
        llm_loader = get_llm_loader()
        available_llms = llm_loader.list_available_llms()
        
        return StatusResponse(
            status="healthy",
            message="System is operational",
            details={
                "documents_indexed": doc_count,
                "available_llms": available_llms,
                "current_llm": llm_loader.config.get("managerLLM"),
                "supported_formats": document_processor.get_supported_extensions()
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload and process documents
    
    Supports: PDF, DOCX, HTML, CSV, XLSX, TXT
    """
    try:
        processed_files = []
        failed_files = []
        
        for file in files:
            try:
                # Save uploaded file
                file_path = os.path.join(UPLOAD_DIR, file.filename)
                
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # Process document
                split_docs = document_processor.process_document(file_path)
                
                # Add to vector store
                vector_store = get_vector_store()
                vector_store.add_documents(split_docs)
                
                processed_files.append({
                    "filename": file.filename,
                    "chunks": len(split_docs),
                    "status": "success"
                })
                
            except Exception as e:
                failed_files.append({
                    "filename": file.filename,
                    "error": str(e),
                    "status": "failed"
                })
        
        # Reinitialize chatbot chain with new documents
        chatbot = get_chatbot()
        chatbot.reinitialize_chain()
        
        return {
            "message": f"Processed {len(processed_files)} file(s)",
            "processed": processed_files,
            "failed": failed_files
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the RAG system
    
    Sends a question and receives an answer based on uploaded documents
    """
    try:
        chatbot = get_chatbot()
        
        # Get response
        response = chatbot.chat(request.question)
        
        # Format source documents
        sources = []
        for doc in response.get("source_documents", []):
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            })
        
        return ChatResponse(
            answer=response["answer"],
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
async def get_history():
    """Get chat history"""
    try:
        chatbot = get_chatbot()
        history = chatbot.get_chat_history()
        
        return {
            "history": history,
            "count": len(history)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
async def clear_memory():
    """Clear conversation memory"""
    try:
        chatbot = get_chatbot()
        chatbot.clear_memory()
        
        return {
            "message": "Conversation memory cleared",
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents")
async def delete_all_documents():
    """Delete all documents from vector store"""
    try:
        vector_store = get_vector_store()
        vector_store.delete_collection()
        
        # Clear uploaded files
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR)
        
        return {
            "message": "All documents deleted",
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search_documents(query: str, k: int = 4):
    """
    Search for relevant documents
    
    Returns relevant document chunks without generating an answer
    """
    try:
        chatbot = get_chatbot()
        results = chatbot.get_relevant_documents(query, k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": float(score)
            })
        
        return {
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
