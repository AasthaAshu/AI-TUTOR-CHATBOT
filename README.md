# 📚 AI Tutor Chatbot

An intelligent AI-powered chatbot that helps you learn from your study materials! Upload PDFs (lecture notes, textbooks, research papers) and have interactive conversations with your documents. Get summaries, ask questions, and receive simplified explanations of complex terms.

## ✨ Features

- **📄 PDF Upload & Processing** - Upload lecture notes, textbooks, or any study materials
- **💬 Interactive Chat** - Ask questions and get context-aware answers from your documents
- **🔍 Semantic Search** - AI-powered search finds relevant information across your materials
- **📝 Smart Summarization** - Get concise summaries of chapters or topics
- **💡 Term Explanations** - Receive simplified explanations of difficult concepts
- **🎯 RAG Architecture** - Retrieval-Augmented Generation for accurate, source-based answers
- **🆓 Free API** - Uses free HuggingFace Inference API

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Python-based interactive web app)
- **LLM Framework**: LangChain
- **AI Model**: Mistral-7B-Instruct (via HuggingFace API)
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector Database**: ChromaDB
- **PDF Processing**: PyPDF2 & pdfplumber

## 📋 Prerequisites

- Python 3.10 or higher
- HuggingFace account (for free API access)

## 🚀 Installation

### 1. Clone or Navigate to Project Directory

```bash
cd ai-tutor-chatbot
```

### 2. Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Get HuggingFace API Key

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a free account if you don't have one
3. Click "New token" and create a token with **read** permissions
4. Copy your API token

### 5. Configure Environment (Optional)

You can either:
- **Option A**: Enter API key directly in the Streamlit app (recommended for first time)
- **Option B**: Create a `.env` file:

```bash
cp .env.example .env
```

Then edit `.env` and add your API key:
```
HUGGINGFACE_API_KEY=hf_your_actual_api_key_here
```

## 🎮 Usage

### Start the Application

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### Using the Chatbot

1. **Enter API Key**: In the sidebar, paste your HuggingFace API key
2. **Upload PDF**: Click "Choose a PDF file" and select your document
3. **Process**: Click "Process PDF" button to analyze the document
4. **Chat**: Start asking questions in the chat interface!

### Example Questions

- "What are the main topics covered in this document?"
- "Can you summarize chapter 3?"
- "Explain what quantum entanglement means in simple terms"
- "What is the difference between supervised and unsupervised learning?"

### Quick Actions

- **📝 Summarize**: Get a quick summary of the entire document
- **🔄 Clear Chat**: Reset the conversation history
- **🗑️ Clear Documents**: Remove all uploaded documents and start fresh

## 📁 Project Structure

```
ai-tutor-chatbot/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
├── src/
│   ├── __init__.py           # Package initializer
│   ├── config.py             # Configuration settings
│   ├── document_processor.py # PDF extraction & chunking
│   ├── vector_store.py       # ChromaDB vector store manager
│   └── llm_handler.py        # LLM interactions
├── data/                      # ChromaDB storage (auto-created)
└── uploads/                   # Temporary PDF storage (auto-created)
```

## 🔧 Configuration

Edit `src/config.py` to customize:

- **Embedding Model**: Change the sentence-transformers model
- **LLM Model**: Switch to different HuggingFace models
- **Chunk Size**: Adjust document chunking parameters
- **Upload Settings**: Change file size limits

## 🐛 Troubleshooting

### API Key Issues
- Make sure your HuggingFace token has read permissions
- Check that you're using a valid token (they start with `hf_`)

### PDF Processing Fails
- Ensure PDF is not password-protected
- Try PDFs with selectable text (not scanned images)
- Check file size is under 10MB

### Model Loading Slow
- First time loading models downloads them (can take a few minutes)
- Subsequent runs will be faster
- Consider using a smaller embedding model if needed

### Memory Issues
- Reduce `CHUNK_SIZE` in `src/config.py`
- Process smaller PDFs
- Close other applications

## 🌟 Features in Detail

### Document Processing
- Extracts text from PDFs using dual approach (pdfplumber + PyPDF2)
- Intelligently chunks documents for optimal retrieval
- Preserves metadata and source information

### Vector Search
- Uses local embeddings (no API calls for embeddings)
- Persistent storage with ChromaDB
- Fast semantic similarity search

### LLM Integration
- Context-aware responses using RAG
- Custom prompts optimized for education
- Source citation for transparency

## 🔮 Future Enhancements

- [ ] Support for multiple file formats (DOCX, TXT, EPUB)
- [ ] Multi-document chat (query across multiple PDFs)
- [ ] Export chat history
- [ ] Flashcard generation
- [ ] Quiz creation from documents
- [ ] Audio explanations (TTS)
- [ ] Dark mode theme

## 📝 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 🙏 Acknowledgments

- HuggingFace for free LLM API access
- LangChain for the excellent framework
- Streamlit for the amazing UI framework
- ChromaDB for vector storage

## 📧 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Verify your HuggingFace API key is valid

---

**Happy Learning! 📚✨**
