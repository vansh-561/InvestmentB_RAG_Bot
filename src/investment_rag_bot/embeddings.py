"""
Embeddings module for investment banking RAG bot.
Handles text embeddings using Google Gemini API.
"""

import google.generativeai as genai
from typing import List#, Union
from .config import config

class GeminiEmbeddings:
    """Interface to Google's Gemini Embeddings API"""
    
    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize Gemini Embeddings.
        
        Args:
            api_key: Google API key (defaults to config)
            model_name: Gemini embedding model name (defaults to config)
        """
        self.api_key = api_key or config.google_api_key
        if not self.api_key:
            raise ValueError("Google API key is required for Gemini Embeddings")
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        self.model_name = model_name or config.gemini_embedding_model
        
        print(f"Initialized Gemini Embeddings with model: {self.model_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            # Truncate text if it's too long (Gemini has token limits)
            if len(text) > 25000:
                print(f"Warning: Truncating text from {len(text)} chars to 25000 chars")
                text = text[:25000]
            
            # Use the embedding generation method directly
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            
            embedding = result["embedding"]
            # Ensure the embedding isn't all zeros for Pinecone
            if all(v == 0 for v in embedding):
                print("Warning: All-zero embedding detected, adding small non-zero value")
                embedding[0] = 1e-5
                
            return embedding
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return a fallback embedding with a small non-zero value
            fallback_embedding = [0.0] * 768 
            fallback_embedding[0] = 1e-5
            return fallback_embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 5) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        results = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Process each text in the batch
            batch_embeddings = []
            for text_item in batch:
                embedding = self.embed_text(text_item)
                batch_embeddings.append(embedding)
            
            results.extend(batch_embeddings)
            
            # Add a small delay between batches to avoid rate limits
            if i + batch_size < len(texts):
                import time
                time.sleep(0.5)
        
        return results
