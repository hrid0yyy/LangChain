import os
from typing import List, Optional
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from src.loader import load_pdf
from src.splitter import split_docs
from src.vector_store import create_vector_store, load_vector_store, vector_store_exists
from src.retriever import retriever
from src.prompt import get_prompt

class RAGQNA:
    """
    A class for Retrieval-Augmented Generation (RAG) question-answering using a vector store and LLM.

    Attributes:
        vector_store: The vector store used for document retrieval.
        retriever_instance: The retriever for fetching relevant documents.
    """

    def __init__(self, pdf_path: str, llm=None):
        """
        Initializes the RAGQNA system by loading a PDF, creating/loading a vector store, and setting up a retriever.

        Args:
            pdf_path: Path to the PDF file to process.
            llm: Optional language model for the retriever. If None, uses default from retriever function.

        Raises:
            ValueError: If PDF loading or document splitting fails.
            FileNotFoundError: If the PDF file does not exist.
            RuntimeError: If vector store creation/loading fails.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Load and process PDF
        docs = load_pdf(pdf_path)
        if not docs:
            raise ValueError("No documents loaded from PDF")

        chunks = split_docs(docs)
        if not chunks:
            raise ValueError("No chunks created from documents")

        # Create vector store if it doesn't exist
        if not vector_store_exists():
            create_vector_store(chunks)

        # Load vector store
        try:
            self.vector_store = load_vector_store()
        except Exception as e:
            raise RuntimeError(f"Failed to load vector store: {str(e)}")

        # Initialize retriever
        try:
            self.retriever_instance = retriever(vectorstore=self.vector_store, llm=llm)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize retriever: {str(e)}")

    def query(self, question: str, template: Optional[str] = None) -> List[Document]:
        """
        Retrieves relevant documents for a given question using the configured retriever.

        Args:
            question: The question to answer.
            template: Optional custom prompt template for context and question.

        Returns:
            A list of Document objects containing relevant information.

        Raises:
            ValueError: If the question is empty or not a string.
            RuntimeError: If retrieval fails.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string")

        try:
            # Get prompt (not used directly in retrieval but included for potential future use)
            get_prompt(context="", question=question, template=template)

            # Retrieve documents
            docs = self.retriever_instance.invoke(question)
            return docs
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve documents: {str(e)}")

    def answer(self, question: str, template: Optional[str] = None) -> str:
        """
        Generates an answer to a question by retrieving relevant documents and formatting them.

        Args:
            question: The question to answer.
            template: Optional custom prompt template for context and question.

        Returns:
            A string containing the formatted answer with retrieved document contents.

        Raises:
            ValueError: If the question is empty or not a string.
            RuntimeError: If retrieval fails.
        """
        docs = self.query(question, template)
        if not docs:
            return "No relevant information found."

        # Format the answer
        answer = f"Answer to: {question}\n\nRelevant Information:\n"
        for i, doc in enumerate(docs, 1):
            answer += f"Document {i}:\n{doc.page_content}\n\n"
        return answer