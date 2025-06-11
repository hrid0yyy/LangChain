from langchain.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configuration
PERSIST_DIRECTORY = "vector_store"
COLLECTION_NAME = "sample"

def initialize_vector_store():
    """Initialize and return a Chroma vector store."""
    try:
        return Chroma(
            embedding_function=MistralAIEmbeddings(model="mistral-embed"),
            persist_directory=PERSIST_DIRECTORY,
            collection_name=COLLECTION_NAME
        )
    except Exception as e:
        raise Exception(f"Failed to initialize vector store: {str(e)}")

def create_vector_store(docs):
    """
    Create a new vector store with the given documents.
    
    Args:
        docs: List of documents to add to the vector store
    """
    try:
        if not docs:
            raise ValueError("Document list is empty")
        
        vector_store = initialize_vector_store()
        vector_store.add_documents(docs)
        return vector_store
    except Exception as e:
        raise Exception(f"Failed to create vector store: {str(e)}")

def load_vector_store():
    """
    Load an existing vector store.
    
    Returns:
        Chroma vector store instance
    """
    try:
        if not os.path.exists(PERSIST_DIRECTORY):
            raise FileNotFoundError(f"Vector store directory {PERSIST_DIRECTORY} does not exist")
        
        return initialize_vector_store()
    except Exception as e:
        raise Exception(f"Failed to load vector store: {str(e)}")

def add_to_vector_store(docs):
    """
    Add documents to an existing vector store.
    
    Args:
        docs: List of documents to add
    """
    try:
        if not docs:
            raise ValueError("Document list is empty")
        
        vector_store = load_vector_store()
        vector_store.add_documents(docs)
        return vector_store
    except Exception as e:
        raise Exception(f"Failed to add documents to vector store: {str(e)}")

def vector_store_exists():
    """Check if vector store exists, return None if it doesn't."""
    return PERSIST_DIRECTORY if os.path.exists(PERSIST_DIRECTORY) else None