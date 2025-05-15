"""
Streamlit app for Investment Banking RAG Bot.
Provides a user interface for document upload and querying.
"""

import os
import tempfile
import streamlit as st
from pathlib import Path
from src.investment_rag_bot.rag_engine import RAGEngine
from src.investment_rag_bot.config import config

# Ensure temp directory exists
TEMP_DIR = Path(tempfile.gettempdir()) / "investment_rag_bot"
TEMP_DIR.mkdir(exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="Investment Banking RAG Bot",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #4B5563;
    }
    .highlight {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563EB;
    }
    .user-message {
        background-color: #00008B;
        color: #FFFFFF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .bot-message {
        background-color: #006400;
        color: #FFFFFF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #60A5FA;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Investment Banking RAG Bot</h1>', unsafe_allow_html=True)
st.markdown('<p class="info-text">Upload investment banking documents and ask questions about them.</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<h2 class="sub-header">Document Upload</h2>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload investment banking PDFs to query"
    )
    
    # Process uploaded files
    if uploaded_files:
        with st.spinner("Processing documents..."):
            for uploaded_file in uploaded_files:
                # Save uploaded file to temp directory
                temp_file_path = os.path.join(TEMP_DIR, uploaded_file.name)
                
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process the file
                success = st.session_state.rag_engine.process_pdf(temp_file_path)
                
                if success:
                    st.success(f"Processed: {uploaded_file.name}")
                else:
                    st.error(f"Failed to process: {uploaded_file.name}")
    
    # Display stats
    st.markdown('<h2 class="sub-header">System Stats</h2>', unsafe_allow_html=True)
    stats = st.session_state.rag_engine.get_stats()
    
    st.markdown(f"**Processed Files:** {stats.get('file_count', 0)}")
    st.markdown(f"**Vector Count:** {stats.get('vector_count', 0)}")
    
    # List processed files
    if stats.get('processed_files'):
        with st.expander("View Processed Files"):
            for file in stats.get('processed_files', []):
                st.markdown(f"- {file}")
    
    # Settings
    st.markdown('<h2 class="sub-header">Settings</h2>', unsafe_allow_html=True)
    
    # Adjust retrieval settings
    config.retrieval_top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=config.retrieval_top_k
    )
    
    config.similarity_threshold = st.slider(
        "Similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=config.similarity_threshold,
        step=0.05
    )

# Main area
st.markdown('<h2 class="sub-header">Ask Questions</h2>', unsafe_allow_html=True)

# Query input
query = st.text_input("Enter your investment banking question:")

# Process query
if query:
    with st.spinner("Processing query..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Get response from RAG engine
        result = st.session_state.rag_engine.query(query)
        
        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": result["response"]})
        
        # Store context for display
        st.session_state.last_context = result.get("context", [])
        st.session_state.processing_time = result.get("processing_time", 0)

# Display chat history
st.markdown('<h2 class="sub-header">Conversation</h2>', unsafe_allow_html=True)

for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)

# Display context for the last query
if st.session_state.chat_history and "last_context" in st.session_state:
    with st.expander("View Retrieved Context"):
        if st.session_state.last_context:
            st.markdown(f"**Processing Time:** {st.session_state.processing_time:.2f} seconds")
            
            for i, context in enumerate(st.session_state.last_context):
                st.markdown(f"### Source {i+1}: {context['source']}")
                st.markdown(f"**Relevance Score:** {context['score']:.4f}")
                st.markdown('<div class="highlight">' + context['text'] + '</div>', unsafe_allow_html=True)
        else:
            st.markdown("No relevant context found for this query.")

# Footer
st.markdown("---")
st.markdown('<p class="info-text">Investment Banking RAG Bot using Google Gemini and Pinecone</p>', unsafe_allow_html=True)
