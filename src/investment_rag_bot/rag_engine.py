"""
RAG Engine module for investment banking RAG bot.
Orchestrates the entire RAG workflow.
"""

import os
import time
from typing import Dict, Any#, Optional
from .config import config
from .pdf_processor import PDFProcessor
from .embeddings import GeminiEmbeddings
from .vector_store import VectorStore
from .llm import GeminiLLM

class RAGEngine:
    """Main RAG engine that orchestrates the entire workflow"""
    
    def __init__(self):
        """Initialize the RAG engine components"""
        # Initialize components
        self.pdf_processor = PDFProcessor(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            timeout=config.pdf_processing_timeout
        )
        self.embeddings = GeminiEmbeddings()
        self.vector_store = VectorStore(dimension=768)  # Gemini embeddings dimension
        self.llm = GeminiLLM()
        
        # Track processed files
        self.processed_files = set()
        
        print("RAG Engine initialized successfully")
    
    def process_pdf(self, pdf_path: str) -> bool:
        """
        Process a PDF file and store its embeddings.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            bool: Success status
        """
        try:
            # Check if file exists
            if not os.path.exists(pdf_path):
                print(f"File not found: {pdf_path}")
                return False
            
            # Check if file has already been processed
            filename = os.path.basename(pdf_path)
            if filename in self.processed_files:
                print(f"File already processed: {filename}")
                return True
            
            print(f"Processing PDF: {filename}")
            start_time = time.time()
            
            # Extract text and chunk it
            chunks = self.pdf_processor.process_pdf(pdf_path)
            if not chunks:
                print(f"No text extracted from {filename}")
                return False
            
            print(f"Generated {len(chunks)} chunks from {filename}")
            
            # Generate embeddings for chunks
            print("Generating embeddings...")
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embeddings.embed_batch(texts)
            
            # Add embeddings to chunks
            for i, embedding in enumerate(embeddings):
                chunks[i]["embedding"] = embedding
                chunks[i]["id"] = f"{filename}_{i}"
            
            # Store in vector database
            print("Storing embeddings in vector database...")
            success = self.vector_store.upsert_items(chunks)
            
            if success:
                self.processed_files.add(filename)
                print(f"Successfully processed {filename} in {time.time() - start_time:.2f} seconds")
                return True
            else:
                print(f"Failed to store embeddings for {filename}")
                return False
                
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return False
    
    def query(self, query_text: str) -> Dict[str, Any]:
        """
        Process a query using the RAG workflow.
        
        Args:
            query_text: User's query
            
        Returns:
            Dict with response and context information
        """
        try:
            start_time = time.time()
            
            # Generate embedding for query
            query_embedding = self.embeddings.embed_text(query_text)
            
            # Retrieve relevant chunks
            results = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=config.retrieval_top_k
            )
            
            # Filter results by similarity threshold
            filtered_results = [
                r for r in results 
                if r["score"] >= config.similarity_threshold
            ]
            
            # Prepare context from retrieved chunks
            if filtered_results:
                context = "\n\n".join([
                    f"[Document: {r['source']}]\n{r['text']}"
                    for r in filtered_results
                ])
                
                # Generate response with context
                response = self.llm.generate_response(query_text, context)
                
                return {
                    "response": response,
                    "context": filtered_results,
                    "has_context": True,
                    "processing_time": time.time() - start_time
                }
            else:
                # No relevant context found
                response = self.llm.generate_response(query_text)
                
                return {
                    "response": response,
                    "context": [],
                    "has_context": False,
                    "processing_time": time.time() - start_time
                }
                
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "response": "I'm sorry, I encountered an error while processing your query.",
                "context": [],
                "has_context": False,
                "error": str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG engine.
        
        Returns:
            Dict with statistics
        """
        try:
            vector_stats = self.vector_store.get_stats()
            
            return {
                "processed_files": list(self.processed_files),
                "file_count": len(self.processed_files),
                "vector_count": vector_stats.get("namespaces", {}).get(
                    config.pinecone_namespace, {}).get("vector_count", 0),
                "index_fullness": vector_stats.get("index_fullness", 0),
                "vector_stats": vector_stats
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {
                "error": str(e)
            }
