# Investment Banking RAG Bot

A Retrieval-Augmented Generation (RAG) bot for investment banking documents using Langchain, Pinecone, and Google Gemini.

## Features

- Drag and drop interface for uploading investment banking documents
- PDF processing and vectorization
- RAG-enhanced responses using Google Gemini LLM
- Streamlit interface for easy interaction

## Setup

1. Clone this repository
2. Install dependencies with Poetry:
```bash
poetry install
```

3. Create a `.env` file with your API keys (see `.env.example`)
4. Run the application:
```bash
poetry run streamlit run run.py
```

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=investment-docs
PINECONE_NAMESPACE=investment-banking
```

## Usage

1. Upload investment banking documents through the Streamlit interface
2. Wait for the documents to be processed
3. Ask questions about the content of your documents

## License

MIT