"""
Configuration settings for AI Tutor Chatbot
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# Model Settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # Free via HuggingFace Inference API

# ChromaDB Settings
CHROMA_PERSIST_DIRECTORY = "./data/chroma_db"
COLLECTION_NAME = "documents"

# Document Processing Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Streamlit Settings
PAGE_TITLE = "AI Tutor - Smart Learning Assistant"
PAGE_ICON = "📚"

# Upload Settings
UPLOAD_DIRECTORY = "./uploads"
MAX_FILE_SIZE_MB = 10
