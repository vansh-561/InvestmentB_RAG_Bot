"""
LLM module for investment banking RAG bot.
Handles interactions with Google Gemini LLM.
"""

import google.generativeai as genai
from typing import Optional
from .config import config

class GeminiLLM:
    """Interface to Google's Gemini LLM API"""
    
    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize Gemini LLM.
        
        Args:
            api_key: Google API key (defaults to config)
            model_name: Gemini model name (defaults to config)
        """
        self.api_key = api_key or config.google_api_key
        if not self.api_key:
            raise ValueError("Google API key is required for Gemini LLM")
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        self.model_name = model_name or config.gemini_llm_model
        
        # Configure generation parameters
        self.generation_config = genai.GenerationConfig(
            temperature=0.2,  # Low temperature for more focused, factual responses
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
        )
        
        # Cache the model to avoid recreating it
        self._model = None
        
        print(f"Initialized Gemini LLM with model: {self.model_name}")
    
    @property
    def model(self):
        """Lazily initialize and cache the generative model"""
        if self._model is None:
            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config
            )
        return self._model
    
    def generate_response(self, query: str, context: Optional[str] = None) -> str:
        """
        Generate a response from Gemini based on a query and optional context.
        
        Args:
            query: User's query
            context: Optional context from retrieved documents
            
        Returns:
            Generated response
        """
        try:
            if context:
                prompt = self._create_rag_prompt(query, context)
            else:
                prompt = self._create_standard_prompt(query)
            
            response = self.model.generate_content(prompt)
            
            return response.text
        except Exception as e:
            print(f"Error generating response from Gemini: {e}")
            return "I'm sorry, I encountered an error while generating a response. Please try again."
    
    def _create_standard_prompt(self, query: str) -> str:
        """
        Create a standard prompt for non-RAG queries.
        
        Args:
            query: User's query
            
        Returns:
            Formatted prompt
        """
        return f"""You are an investment banking assistant based on specific investment literature and financial documents.
        
Question: {query}

If this question is outside the scope of your investment banking knowledge, please respond with:
"I'm sorry, but I don't have information about that in my investment documentation."

Otherwise, provide a clear, accurate, and helpful response to the investment banking question."""
    
    def _create_rag_prompt(self, query: str, context: str) -> str:
        """
        Create a RAG-enhanced prompt with context from retrieved documents.
        
        Args:
            query: User's query
            context: Context from retrieved documents
            
        Returns:
            Formatted prompt
        """
        return f"""You are an investment banking assistant based on specific investment literature and financial documents.
        
Below is information from the investment banking documents that may be relevant to the question:

{context}

Question: {query}

Based ONLY on the information provided above, please answer the question. 
If the information provided doesn't contain the answer, respond with:
"I'm sorry, but I don't have information about that in my investment documentation."

Your response should be:
1. Accurate and based only on the provided context
2. Clear and easy to understand
3. Properly formatted for readability
4. Concise but thorough"""