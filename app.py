"""
AI Tutor Chatbot - Main Streamlit Application
Interactive chatbot for learning with PDF documents
"""
import os
import streamlit as st
from pathlib import Path
from src.config import PAGE_TITLE, PAGE_ICON, UPLOAD_DIRECTORY, MAX_FILE_SIZE_MB
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.llm_handler import LLMHandler


# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "llm_handler" not in st.session_state:
    st.session_state.llm_handler = None

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

if "current_file" not in st.session_state:
    st.session_state.current_file = None


def initialize_components(api_key: str):
    """Initialize vector store and LLM handler"""
    try:
        if st.session_state.vector_store is None:
            st.session_state.vector_store = VectorStoreManager()
        
        if st.session_state.llm_handler is None:
            st.session_state.llm_handler = LLMHandler(api_key=api_key)
        
        return True
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        return False


def process_pdf(uploaded_file, api_key):
    """Process uploaded PDF file"""
    try:
        # Ensure upload directory exists
        os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
        
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Initialize components
        if not initialize_components(api_key):
            return False
        
        # Process document
        with st.spinner("Processing PDF... This may take a moment."):
            doc_processor = DocumentProcessor()
            chunks = doc_processor.process_pdf(file_path, uploaded_file.name)
            
            # Add to vector store
            success = st.session_state.vector_store.add_documents(chunks)
            
            if success:
                st.session_state.documents_loaded = True
                st.session_state.current_file = uploaded_file.name
                st.success(f"✅ Successfully processed {len(chunks)} chunks from {uploaded_file.name}")
                return True
            else:
                st.error("Failed to add documents to vector store")
                return False
                
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return False


def main():
    """Main application"""
    
    # Header
    st.title(f"{PAGE_ICON} AI Tutor - Smart Learning Assistant")
    st.markdown("Upload your lecture notes, textbooks, or study materials and chat with them!")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Setup")
        
        # API Key input
        api_key = st.text_input(
            "HuggingFace API Key",
            type="password",
            help="Get your free API key from https://huggingface.co/settings/tokens"
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your HuggingFace API key to continue")
            st.info("👉 Get a free API key at: https://huggingface.co/settings/tokens")
        
        st.divider()
        
        # File upload
        st.header("📄 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB"
        )
        
        if uploaded_file and api_key:
            if st.button("Process PDF", type="primary"):
                process_pdf(uploaded_file, api_key)
        
        # Document info
        if st.session_state.documents_loaded and st.session_state.current_file:
            st.success(f"✅ Loaded: {st.session_state.current_file}")
        
        st.divider()
        
        # Quick actions
        st.header("🎯 Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📝 Summarize", disabled=not st.session_state.documents_loaded):
                if st.session_state.documents_loaded:
                    st.session_state.messages.append({
                        "role": "user",
                        "content": "Please provide a summary of the key topics in this document."
                    })
                    st.rerun()
        
        with col2:
            if st.button("🔄 Clear Chat", disabled=len(st.session_state.messages) == 0):
                st.session_state.messages = []
                st.rerun()
        
        # Clear documents
        if st.button("🗑️ Clear All Documents", disabled=not st.session_state.documents_loaded):
            if st.session_state.vector_store:
                st.session_state.vector_store.clear_vectorstore()
                st.session_state.documents_loaded = False
                st.session_state.current_file = None
                st.session_state.messages = []
                st.success("All documents cleared!")
                st.rerun()
    
    # Main chat area
    if not api_key:
        st.info("👈 Please enter your HuggingFace API key in the sidebar to get started")
        return
    
    if not st.session_state.documents_loaded:
        st.info("👈 Please upload a PDF document in the sidebar to begin chatting")
        
        # Show features
        st.subheader("✨ Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📚 Document Chat")
            st.write("Upload PDFs and ask questions about the content")
        
        with col2:
            st.markdown("### 🔍 Smart Search")
            st.write("AI finds relevant information from your documents")
        
        with col3:
            st.markdown("### 💡 Term Explanations")
            st.write("Get simplified explanations of difficult concepts")
        
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get retriever
                    retriever = st.session_state.vector_store.get_retriever(k=3)
                    
                    if retriever:
                        # Get relevant documents
                        docs = st.session_state.vector_store.similarity_search(prompt, k=3)
                        
                        # Generate response with context
                        response = st.session_state.llm_handler.chat_with_context(
                            prompt, 
                            docs
                        )
                        
                        st.markdown(response)
                        
                        # Show sources
                        if docs:
                            with st.expander("📚 View Sources"):
                                for i, doc in enumerate(docs, 1):
                                    st.markdown(f"**Source {i}:** {doc.metadata.get('source', 'Unknown')}")
                                    st.text(doc.page_content[:300] + "...")
                                    st.divider()
                    else:
                        response = "I don't have any documents loaded yet. Please upload a PDF first."
                        st.markdown(response)
                    
                    # Add assistant response to chat
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
