"""
Vector store management using FAISS for document embeddings and retrieval
"""
import os
import pickle
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import (
    FAISS_PERSIST_DIRECTORY,
    EMBEDDING_MODEL
)


class VectorStoreManager:
    """Manages vector embeddings and similarity search using FAISS"""
    
    def __init__(self):
        # Initialize embeddings model (runs locally, no API key needed)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu', 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )
        
        # Ensure persist directory exists
        os.makedirs(FAISS_PERSIST_DIRECTORY, exist_ok=True)
        
        # Initialize or load existing FAISS
        self.vectorstore = None
        self._load_or_create_vectorstore()
    
    def _load_or_create_vectorstore(self):
        """Load existing vector store or create new one"""
        try:
            # Try to load existing vectorstore
            index_path = os.path.join(FAISS_PERSIST_DIRECTORY, "index.faiss")
            if os.path.exists(index_path):
                self.vectorstore = FAISS.load_local(
                    FAISS_PERSIST_DIRECTORY,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                self.vectorstore = None
        except Exception as e:
            print(f"Creating new vector store: {e}")
            self.vectorstore = None
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to the vector store
        Returns True if successful
        """
        try:
            if self.vectorstore is None:
                # Create new vectorstore
                self.vectorstore = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
            else:
                # Add to existing vectorstore
                self.vectorstore.add_documents(documents)
            
            # Persist changes
            self.vectorstore.save_local(FAISS_PERSIST_DIRECTORY)
            return True
            
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Search for similar documents based on query
        Returns top k most relevant document chunks
        """
        if self.vectorstore is None:
            return []
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 3) -> List[tuple]:
        """
        Search with relevance scores
        Returns list of (Document, score) tuples
        """
        if self.vectorstore is None:
            return []
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"Error during similarity search with score: {e}")
            return []
    
    def clear_vectorstore(self):
        """Clear all documents from the vector store"""
        try:
            self.vectorstore = None
            
            # Clear the persist directory
            import shutil
            if os.path.exists(FAISS_PERSIST_DIRECTORY):
                shutil.rmtree(FAISS_PERSIST_DIRECTORY)
                os.makedirs(FAISS_PERSIST_DIRECTORY, exist_ok=True)
            
            return True
        except Exception as e:
            print(f"Error clearing vector store: {e}")
            return False
    
    def get_retriever(self, k: int = 3):
        """
        Get a LangChain retriever for use in chains
        """
        if self.vectorstore is None:
            return None
        
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
