# Document Processing and RAG System

## Overview
A sophisticated document processing and retrieval system that supports multiple file types and provides intelligent question-answering capabilities.

Tesseract is an open source optical character recognition (OCR) platform. OCR extracts text from images and documents without a text layer and outputs 
the document into a new searchable text file, PDF, or most other popular formats.
## Key Functions

### Document Processing
- `load_documents(directory)`: 
  - Scans a directory for documents
  - Supports PDF, TXT, CSV, image files
  - Extracts text content from various file types

- `DocumentProcessor` Class Methods:
  - `process_image()`: OCR text extraction from images
  - `load_text_file()`: Read text files
  - `load_pdf_file()`: Extract text from PDFs
  - `load_csv_file()`: Convert CSV to documents

### Retrieval and Ranking
- `create_vectorstore()`: 
  - Converts documents into vector embeddings
  - Supports FAISS and Chroma vector stores
  - Splits documents into manageable chunks

- `create_reranking_retriever()`: 
  - Uses Cohere's reranking to improve document relevance
  - Filters and ranks retrieved documents

### Relevance and Filtering
- `get_most_relevant_file()`: 
  - Determines most relevant file for a query
  - Uses semantic similarity and file type weights

- `filter_documents_by_file()`: 
  - Filters documents from a specific file
  - Useful for focused searches

- `filter_documents_by_relevance()`: 
  - Ranks documents by relevance to a query
  - Returns top K most relevant documents

- `validate_context_relevance()`: 
  - Checks if retrieved documents match query
  - Ensures high-quality context for responses

### RAG Response Generation
- `initialize_rag()`: 
  - Sets up RAG pipeline
  - Configures retriever, language model, and prompt

- `rag_response()`: 
  - Main entry point for question answering
  - Retrieves and generates contextual responses

- `format_response()`: 
  - Structures RAG system output
  - Provides detailed response with sources

## Requirements
- Python 3.8+
- Libraries: 
  - langchain
  - openai
  - cohere
  - pandas
  - pytesseract
  - PIL

## Configuration
- Set `OPENAI_API_KEY`
- Set `COHERE_API_KEY`
- Configure `DOCS_DIR` for document source

## Usage
```python
response = rag_response("Your question here")
print(response['Answer'])
```

## Features
- Multi-file type support
- Semantic search
- Contextual question answering
- Source document tracking
- Robust error handling