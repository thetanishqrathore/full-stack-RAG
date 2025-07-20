# app/services.py

import os
import logging
from fastapi import HTTPException

# LangChain & AI Imports
import langchain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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
try:
    llm = ChatOllama(model="mistral:7b-instruct-q4_K_M")
    logger.info("Local LLM (Ollama) initialized successfully.")

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in .env file.")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    logger.info("Google Embedding Model initialized successfully.")
    
    vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    logger.info("ChromaDB Vector Store initialized successfully.")
except Exception as e:
    logger.error(f"Error during AI/DB initialization: {e}")
    raise

# --- SERVICE FUNCTIONS ---

def get_document_loader(file_path: str):
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    if extension == ".pdf": return PyPDFLoader(file_path)
    if extension == ".docx": return Docx2txtLoader(file_path)
    if extension == ".csv": return CSVLoader(file_path)
    if extension in [".html", ".htm"]: return BSHTMLLoader(file_path)
    return UnstructuredFileLoader(file_path)

def process_uploaded_file(file_path: str, filename: str):
    try:
        loader = get_document_loader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        vector_store.add_documents(chunks)
        logger.info(f"Successfully added {filename} to the knowledge base.")
    except Exception as e:
        logger.error(f"Error processing file {filename}: {e}")
        raise e

def get_indexed_files():
    try:
        items = vector_store._collection.get(include=["metadatas"])
        if not items['ids']: return []
        sources = [meta.get('source') for meta in items['metadatas'] if meta]
        unique_files = sorted(list(set(os.path.basename(s) for s in sources if s)))
        return unique_files
    except Exception as e:
        logger.error(f"Could not retrieve files from ChromaDB: {e}")
        return []

def delete_documents_by_source(filename: str):
    try:
        all_docs = vector_store._collection.get(include=["metadatas"])
        ids_to_delete = [
            doc_id for i, doc_id in enumerate(all_docs['ids'])
            if all_docs['metadatas'][i] and os.path.basename(all_docs['metadatas'][i].get('source', '')) == filename
        ]
        if not ids_to_delete:
            logger.warning(f"No documents found for source: {filename}.")
            return
        vector_store._collection.delete(ids=ids_to_delete)
        logger.info(f"Successfully deleted documents for source: {filename}")
    except Exception as e:
        logger.error(f"Error deleting documents for source {filename}: {e}")
        raise e

def get_rag_response_stream(input_text: str, chat_history: list):
    """Gets a streaming response from the RAG chain using a robust, standard pattern."""
    
    history_messages = []
    for human, ai in chat_history:
        history_messages.append(HumanMessage(content=human))
        history_messages.append(AIMessage(content=ai))

    retriever = vector_store.as_retriever()
    
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

    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, just say "
        "that you don't know. Be concise."
        "\n\nContext:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain.stream({
        "input": input_text,
        "chat_history": history_messages
    })
