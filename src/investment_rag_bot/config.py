"""
Configuration module for investment banking RAG bot.
"""

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

class RagConfig(BaseModel):
    """Configuration settings for the RAG application"""
    
    # API Keys
    google_api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    pinecone_api_key: str = Field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    
    # Pinecone settings
    pinecone_index_name: str = Field(
        default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "investment-docs")
    )
    pinecone_namespace: str = Field(
        default_factory=lambda: os.getenv("PINECONE_NAMESPACE", "investment-banking")
    )
    
    # Google Gemini settings
    gemini_embedding_model: str = "models/embedding-001"
    gemini_llm_model: str = "gemini-1.5-flash"
    
    # PDF processing settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    pdf_processing_timeout: int = 300  # 5 minutes
    
    # RAG settings
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.75

    def validate_api_keys(self) -> tuple[bool, str]:
        """Validate that necessary API keys are present"""
        missing_keys = []
        
        if not self.google_api_key:
            missing_keys.append("GOOGLE_API_KEY")
        
        if not self.pinecone_api_key:
            missing_keys.append("PINECONE_API_KEY")
        
        if missing_keys:
            return False, f"Missing required API keys: {', '.join(missing_keys)}"
        
        return True, "All API keys are present"


# Create a global instance of the configuration
config = RagConfig()