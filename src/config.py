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
LLM_MODEL = "google/flan-t5-base"  # Reliable free model via HuggingFace Inference API

# FAISS Settings
FAISS_PERSIST_DIRECTORY = "./data/faiss_db"

# Document Processing Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Streamlit Settings
PAGE_TITLE = "AI Tutor - Smart Learning Assistant"
PAGE_ICON = "📚"

# Upload Settings
UPLOAD_DIRECTORY = "./uploads"
MAX_FILE_SIZE_MB = 10
