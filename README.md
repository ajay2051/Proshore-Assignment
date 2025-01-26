# 01 Document Processing and RAG System

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


# 02 Database Query Agent

## Overview
A sophisticated Python-based agent for natural language database querying using LangChain and OpenAI.

## Features
- Natural language to SQL query generation
- Error-robust database interaction
- Conversational result interpretation
- Comprehensive logging and error handling

## Prerequisites
- Python 3.8+
- PostgreSQL database
- OpenAI API key
- Required Python packages (see `requirements.txt`)

## Environment Variables
Configure these environment variables:
- `DB_USER`: Database username
- `DB_PASS`: Database password
- `DB_HOST`: Database host
- `DB_NAME`: Database name
- `OPENAI_API_KEY`: OpenAI API key

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python

response = ask_agent("How many questions are in the database?")
print(response)
```

## Logging
Logs are configured with timestamp, log level, and message details.

## Error Handling
- Validates configuration parameters
- Provides user-friendly error messages
- Logs errors for debugging


# 03 Real-Time Object Detection Script

## Overview
Python-based object detection script using TensorFlow Lite and OpenCV, supporting Raspberry Pi and webcam integration.

## Features
- Real-time object detection
- Multi-camera support
- FPS tracking
- Configurable detection parameters

## Prerequisites
- Python 3.7+
- OpenCV
- Picamera2
- TensorFlow Lite
- Raspberry Pi or Linux system

## Installation
```bash
pip install opencv-python picamera2 tflite-support
```

## Configuration
Adjust parameters in script:
- Model path
- Camera settings
- Detection thresholds

## Usage
```bash
python 03_object_detection.py
```
Press 'q' to quit

## Supported Cameras
- Raspberry Pi Camera
- USB Webcam

## Performance
- Configurable threads
- Adaptive FPS calculation
- Low-latency detection

## Troubleshooting
- Check camera connections
- Verify model file path
- Ensure required libraries installed


# 04 Text Generation Model

## Overview
Neural network for predicting next words and generating text using LSTM

## Features
- Word prediction
- Text generation
- Uses news dataset
- Supports creativity parameter

## Requirements
- Python 3.8+
- TensorFlow
- Keras
- NLTK
- Pandas
- NumPy

## Installation
```bash
pip install tensorflow nltk pandas numpy
```

## Usage
```python
# Predict next words
predict_next_word("trump", 5)

# Generate text
generate_text("trump", 100, creativity=3)
```

## Model Details
- Input: 10-word sequences
- Architecture: LSTM + Dense layers
- Trained on news text
- Saves model as "predict_next_word.h5"

## Parameters
- `input_text`: Starting text
- `text_length`: Generated text length
- `creativity`: Word selection randomness

## Limitations
- Requires significant computational resources
- Text quality depends on training data