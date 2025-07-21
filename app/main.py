# app/main.py

import os
import shutil
import logging

# --- CRITICAL FIX for gRPC DNS issues in some environments ---
os.environ['GRPC_DNS_RESOLVER'] = 'native'
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from dotenv import load_dotenv

# --- CRITICAL: Load environment variables FIRST ---
load_dotenv()

# --- REFACTORED IMPORTS ---
# Use absolute imports for clarity within the application package
from . import services
from .models import AskRequest

# --- CONFIGURATION & INITIALIZATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define project directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(ROOT_DIR, "frontend")
UPLOAD_DIR = os.path.join(ROOT_DIR, "uploaded_files")
# Ensure the directory for temporary uploads exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Streaming RAG Chatbot Backend", version="5.1.0")

# Configure CORS to allow all origins, which is useful for development.
# For production, you should restrict this to your frontend's domain.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"])

# --- API ENDPOINTS ---

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Handles file uploads, processes them for the RAG knowledge base,
    and then cleans up the uploaded file.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        # Save the uploaded file temporarily
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the file to add it to the vector store
        services.process_uploaded_file(file_path, file.filename)
        
        return JSONResponse(content={"filename": file.filename, "status": "File processed successfully."})
    except Exception as e:
        logger.error(f"API Error processing file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file after processing
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/api/ask")
async def ask_question(request: AskRequest):
    """
    Handles user questions by getting a streaming response from the RAG chain.
    """
    question = request.q
    chat_history = request.chat_history or []
    logger.info(f"Received question: '{question}' with history length: {len(chat_history)}")

    async def stream_generator():
        try:
            # Get the streaming generator from our service layer
            response_stream = services.get_rag_response_stream(question, chat_history)
            
            # We only want to stream the 'answer' part of the response chunks
            for chunk in response_stream:
                if "answer" in chunk:
                    content = chunk["answer"]
                    logger.debug(f"Streaming chunk: {content}")
                    yield content
        except Exception as e:
            logger.error(f"Error during stream generation: {e}")
            # Yield a final error message if something goes wrong during the stream
            yield f"Error: {str(e)}"

    return StreamingResponse(stream_generator(), media_type="text/plain")

@app.get("/api/files")
async def get_files():
    """Returns a list of all files currently in the knowledge base."""
    try:
        return {"files": services.get_indexed_files()}
    except Exception as e:
        logger.error(f"API Error retrieving files: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve file list.")

@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """Deletes a file and its associated data from the knowledge base."""
    logger.info(f"Received request to delete file: {filename}")
    try:
        services.delete_documents_by_source(filename)
        return JSONResponse(content={"filename": filename, "status": "File deleted successfully."})
    except Exception as e:
        logger.error(f"API Error deleting file {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
# Mount the 'frontend' directory to serve static files like index.html, images, etc.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
