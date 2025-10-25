"""
LLM integration for chat, summarization, and explanations using HuggingFace API
"""
from typing import List
from langchain.schema import Document
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.config import HUGGINGFACE_API_KEY, LLM_MODEL


class LLMHandler:
    """Handles LLM interactions for chat, summarization, and explanations"""
    
    def __init__(self, api_key: str = None):
        """Initialize LLM with HuggingFace API"""
        self.api_key = api_key or HUGGINGFACE_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "HuggingFace API key not found. "
                "Please set HUGGINGFACE_API_KEY in .env file"
            )
        
        # Initialize HuggingFace LLM
        self.llm = HuggingFaceHub(
            repo_id=LLM_MODEL,
            huggingfacehub_api_token=self.api_key,
            model_kwargs={
                "temperature": 0.7,
                "max_length": 512,
                "top_p": 0.95
            }
        )
    
    def create_qa_chain(self, retriever):
        """
        Create a RetrievalQA chain for document Q&A
        """
        # Custom prompt for educational context
        prompt_template = """You are an AI tutor assistant. Use the following context from the student's documents to answer their question. 
If you don't know the answer based on the context, say so. Provide clear, educational explanations.

Context: {context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return qa_chain
    
    def answer_question(self, question: str, retriever) -> dict:
        """
        Answer a question using retrieved context
        Returns dict with 'answer' and 'source_documents'
        """
        try:
            qa_chain = self.create_qa_chain(retriever)
            result = qa_chain({"query": question})
            return {
                "answer": result["result"],
                "sources": result["source_documents"]
            }
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": []
            }
    
    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """
        Summarize given text using LLM
        """
        prompt = f"""Summarize the following text in a clear and concise way (max {max_length} words):

{text[:3000]}  

Summary:"""
        
        try:
            summary = self.llm(prompt)
            return summary.strip()
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def explain_term(self, term: str, context: str = "") -> str:
        """
        Provide a simplified explanation of a difficult term or concept
        """
        if context:
            prompt = f"""You are an AI tutor. Explain the term "{term}" in simple, easy-to-understand language.

Context from the document: {context[:500]}

Provide a clear, beginner-friendly explanation:"""
        else:
            prompt = f"""You are an AI tutor. Explain the term "{term}" in simple, easy-to-understand language suitable for students.

Explanation:"""
        
        try:
            explanation = self.llm(prompt)
            return explanation.strip()
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a general response for any prompt
        """
        try:
            response = self.llm(prompt)
            return response.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def chat_with_context(self, user_message: str, context_docs: List[Document]) -> str:
        """
        Generate a chat response with document context
        """
        # Build context from documents
        context = "\n\n".join([doc.page_content for doc in context_docs[:3]])
        
        prompt = f"""You are an AI tutor helping a student understand their course materials. 
Based on the following context from their documents, respond to their message in a helpful, educational way.

Context:
{context}

Student: {user_message}

Tutor:"""
        
        try:
            response = self.llm(prompt)
            return response.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"
