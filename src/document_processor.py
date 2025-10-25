"""
Document processing utilities for PDF extraction and text chunking
"""
import os
from typing import List
import PyPDF2
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentProcessor:
    """Handles PDF processing and text chunking"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file using pdfplumber (better for complex PDFs)
        Falls back to PyPDF2 if pdfplumber fails
        """
        text = ""
        
        try:
            # Try pdfplumber first (better for tables and complex layouts)
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
        except Exception as e:
            print(f"pdfplumber failed, trying PyPDF2: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
            except Exception as e:
                raise Exception(f"Failed to extract text from PDF: {e}")
        
        return text.strip()
    
    def chunk_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Split text into chunks using RecursiveCharacterTextSplitter
        Returns list of LangChain Document objects
        """
        if metadata is None:
            metadata = {}
        
        # Create documents with metadata
        documents = [Document(page_content=text, metadata=metadata)]
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        return chunks
    
    def process_pdf(self, pdf_path: str, filename: str) -> List[Document]:
        """
        Complete pipeline: extract text from PDF and chunk it
        """
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text:
            raise ValueError("No text could be extracted from the PDF")
        
        # Chunk with metadata
        metadata = {
            "source": filename,
            "file_path": pdf_path
        }
        
        chunks = self.chunk_text(text, metadata)
        
        return chunks
    
    def summarize_text(self, text: str, max_length: int = 500) -> str:
        """
        Create a simple extractive summary (first sentences up to max_length)
        For better summarization, use LLM
        """
        if len(text) <= max_length:
            return text
        
        # Simple extractive summary: return first max_length characters
        summary = text[:max_length].rsplit('.', 1)[0] + '.'
        return summary
