# app/services.py

import os
import logging
from fastapi import HTTPException
import chromadb
# LangChain & AI Imports
import langchain
from langchain_community.embeddings import OllamaEmbeddings # <-- IMPORT THIS
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    UnstructuredFileLoader, PyPDFLoader, Docx2txtLoader, BSHTMLLoader, CSVLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage


# --- CONFIGURATION ---
logger = logging.getLogger(__name__)
# --- FIX: Set debug mode to False for cleaner terminal output ---
langchain.debug = False

# --- DIRECTORIES ---
CHROMA_DB_DIR = "chroma_db"

# --- INITIALIZE MODELS AND VECTOR STORE ---
'''
# for local hosting
try:
    # Initialize the local LLM using Ollama
    llm = ChatOllama(model="mistral:7b-instruct-q4_K_M")
    logger.info("Local LLM (Ollama) initialized successfully.")

    # Initialize Google Generative AI embeddings
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in .env file.")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    logger.info("Google Embedding Model initialized successfully.")
    
    # Initialize ChromaDB vector store for persistence
    vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    logger.info("ChromaDB Vector Store initialized successfully.")
except Exception as e:
    logger.error(f"Error during AI/DB initialization: {e}")
    raise
'''

try:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))

    # Initialize the LLM to connect to the Ollama service
    llm = ChatOllama(model="llama3:8b", base_url=OLLAMA_BASE_URL)
    logger.info(f"Connecting to Ollama at {OLLAMA_BASE_URL}")

    # Initialize Ollama Embeddings for nomic-embed-text
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
    logger.info("Ollama Embedding Model (nomic-embed-text) initialized successfully.")

    # Initialize the ChromaDB client to connect to the ChromaDB server
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    vector_store = Chroma(
        client=chroma_client,
        collection_name="rag_collection",
        embedding_function=embeddings,
    )
    logger.info(f"ChromaDB client connected to http://{CHROMA_HOST}:{CHROMA_PORT}")

except Exception as e:
    logger.error(f"Error during AI/DB initialization: {e}")
    raise


# --- SERVICE FUNCTIONS ---

def get_document_loader(file_path: str):
    """Returns the appropriate document loader based on file extension."""
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    if extension == ".pdf": return PyPDFLoader(file_path)
    if extension == ".docx": return Docx2txtLoader(file_path)
    if extension == ".csv": return CSVLoader(file_path)
    if extension in [".html", ".htm"]: return BSHTMLLoader(file_path)
    # Default loader for text-based files
    return UnstructuredFileLoader(file_path)

def process_uploaded_file(file_path: str, filename: str):
    """Loads, splits, and embeds a document into the vector store."""
    try:
        loader = get_document_loader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        # Add document chunks to the vector store
        vector_store.add_documents(chunks)
        logger.info(f"Successfully added {filename} to the knowledge base.")
    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}")
        raise e

def get_indexed_files():
    """Retrieves a list of unique source filenames from the vector store."""
    try:
        # Get all items and their metadata from the collection
        items = vector_store._collection.get(include=["metadatas"])
        if not items['ids']: return []
        # Extract the 'source' from metadata
        sources = [meta.get('source') for meta in items['metadatas'] if meta]
        # Get a unique, sorted list of base filenames
        unique_files = sorted(list(set(os.path.basename(s) for s in sources if s)))
        return unique_files
    except Exception as e:
        logger.error(f"Could not retrieve files from ChromaDB: {e}")
        return []

def delete_documents_by_source(filename: str):
    """Deletes all document chunks associated with a specific source filename."""
    try:
        all_docs = vector_store._collection.get(include=["metadatas"])
        # Find all document IDs that match the source filename
        ids_to_delete = [
            doc_id for i, doc_id in enumerate(all_docs['ids'])
            if all_docs['metadatas'][i] and os.path.basename(all_docs['metadatas'][i].get('source', '')) == filename
        ]
        if not ids_to_delete:
            logger.warning(f"No documents found for source: {filename}.")
            return
        # Delete the documents from the collection by their IDs
        vector_store._collection.delete(ids=ids_to_delete)
        logger.info(f"Successfully deleted documents for source: {filename}")
    except Exception as e:
        logger.error(f"Error deleting documents for source {filename}: {e}")
        raise e

def get_rag_response_stream(input_text: str, chat_history: list):
    """
    Gets a streaming response from the RAG chain with improved prompting.
    """
    
    # Convert the chat history to the expected format of Human/AI messages
    history_messages = []
    for human, ai in chat_history:
        history_messages.append(HumanMessage(content=human))
        history_messages.append(AIMessage(content=ai))

    retriever = vector_store.as_retriever()
    
    # This prompt helps the model reformulate the user's question to be standalone,
    # using the context of the chat history.
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # *** IMPROVED PROMPT ***
    # This new prompt gives the LLM clearer instructions on how to behave based on the
    # presence and quality of the retrieved context.
    qa_system_prompt = (
        "You are a helpful and polite assistant for question-answering tasks. Your goal is to provide the most accurate and relevant answer possible.\n\n"
        "Here is some context retrieved from a knowledge base that might be relevant to the user's question:\n"
        "----------------\n"
        "Context:\n{context}\n"
        "----------------\n\n"
        "Please follow these rules when answering:\n"
        "1. **Analyze the Context:** Carefully examine the provided context. If it is highly relevant and directly answers the user's question, use it as your primary source. Synthesize the information from the context to form a comprehensive answer.\n"
        "2. **Handle Insufficient Context:** If the context is not relevant, is incomplete, or does not seem to answer the question, DO NOT mention the context or say you couldn't find information. Instead, rely on your own general knowledge to answer the question as helpfully as you can. You can politely state that you are answering based on your general understanding if it feels natural.\n"
        "3. **Prioritize Semantic Meaning:** Focus on the underlying meaning of the question and the context, not just keyword matching.\n"
        "4. **Be Polite and Conversational:** Always maintain a friendly and respectful tone.\n"
        "5. **No Context Provided:** If no context is provided at all (the context block is empty), simply answer the question using your own extensive knowledge base."
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # This chain combines the retrieved documents into a single string.
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # This is the final chain that orchestrates the retrieval and answer generation.
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Stream the response back to the user.
    return rag_chain.stream({
        "input": input_text,
        "chat_history": history_messages
    })
