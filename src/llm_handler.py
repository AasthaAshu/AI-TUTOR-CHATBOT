"""
LLM integration for chat, summarization, and explanations using Groq API
"""
from typing import List
from langchain_core.documents import Document
import os
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class LLMHandler:
    """Handles LLM interactions using Groq API (fast and free)"""
    
    def __init__(self, api_key: str = None):
        """Initialize Groq API client"""
        if not GROQ_AVAILABLE:
            raise ImportError(
                "Groq package not found. Install with: pip install groq"
            )
        
        self.api_key = api_key
        if not self.api_key:
            raise ValueError(
                "Groq API key required. Get free key at: https://console.groq.com"
            )
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.1-8b-instant"  # Fast and accurate
    
    def _generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        """Helper method to generate text using Groq"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"
    
    # def answer_question(self, question: str, context_docs: List[Document]) -> str:
    #     """
    #     Answer a question using retrieved context
    #     Returns the answer string
    #     """
    #     # Build context from documents
    #     context = "\n\n".join([doc.page_content for doc in context_docs[:5]])
        
    #     system = "You are an AI tutor. Answer questions ONLY based on the provided context from the student's document. If the context doesn't contain the answer, say so."
    #     user = f"""Context from the document:
    #     {context[:3000]}

    #      Question: {question}

    #      Answer based ONLY on the context above:"""
        
    #     return self._generate(system, user, max_tokens=400)
    
    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """
        Summarize given text using LLM
        """
        system = "You are an AI tutor. Provide a clear, comprehensive summary of the key topics and main points."
        user = f"""Summarize the key topics and main ideas from this text:

        {text[:4000]}

        Provide a structured summary with the main points:"""
        
        return self._generate(system, user, max_tokens=400)
    
    def explain_term(self, term: str, context: str = "") -> str:
        """
        Provide a simplified explanation of a difficult term or concept
        """
        system = "You are an AI tutor. Explain concepts in simple, clear language."
        if context:
            user = f"Explain '{term}' using this context:\n{context[:800]}"
        else:
            user = f"Explain what '{term}' means in simple terms."
        
        return self._generate(system, user, max_tokens=200)
    
    # def generate_response(self, prompt: str) -> str:
    #     """
    #     Generate a general response for any prompt
    #     """
    #     system = "You are a helpful AI tutor assistant."
    #     return self._generate(system, prompt, max_tokens=300)
    
    def chat_with_context(self, user_message: str, context_docs: List[Document]) -> str:
        """
        Generate a chat response with document context
        """
        # Build context from documents - use more for comprehensive queries
        num_docs = 8 if any(word in user_message.lower() for word in ['all', 'list', 'laws', 'chapter']) else 5
        context = "\n\n".join([doc.page_content for doc in context_docs[:num_docs]])
        
        system = """You are an AI tutor helping students understand their course materials. 
        Answer ONLY based on the provided context from the student's document. 
        Be specific and accurate. If asked to list multiple items (like "all laws" or "4 laws"), make sure to find and list ALL of them from the context.
        If the context doesn't contain complete information, acknowledge what's missing."""
        user = f"""Context from the document:
        {context[:4000]}

        Student question: {user_message}

        Answer based on the context above:"""
        
        return self._generate(system, user, max_tokens=500)
