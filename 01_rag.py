import logging
import os
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pytesseract
from langchain.chains.conversational_retrieval.base import \
    ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not OPENAI_API_KEY or not COHERE_API_KEY:
    raise ValueError("OPENAI_API_KEY or COHERE_API_KEY must be set")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DOCS_DIR = os.getenv("DOCS_DIR", UPLOAD_DIR)
os.makedirs(DOCS_DIR, exist_ok=True)


LLM_CONFIG = {
    "model": "gpt-4o",
    "api_key": OPENAI_API_KEY,
}
llm = ChatOpenAI(**LLM_CONFIG)

class DocumentProcessor:
    """Handle document processing and loading"""

    @staticmethod
    def process_image(image_path: str) -> Optional[Document]:
        """Process Image file using Tesseract OCR"""
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            if text.strip():
                return Document(page_content=text, metadata={"source": image_path, "type": "image", "filename": Path(image_path).name})
            return None
        except FileNotFoundError:
            logger.error(f"File not found: {image_path}")
            return None
        except Exception as e:
            logging.error(f"Error processing Image file {image_path}: {str(e)}")
            return None

    @staticmethod
    def load_text_file(file_path: str) -> Optional[Document]:
        """Load text file"""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                if text:
                    return Document(page_content=text, metadata={"source": file_path, "type": "text", "filename": Path(file_path).name})
                return None
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return None
        except Exception as e:
            logging.error(f"Error processing Text file {file_path}: {str(e)}")
            return None

    @staticmethod
    def load_pdf_file(file_path: str) -> List[Document]:
        """Load PDF file"""
        try:
            loader = PyPDFLoader(file_path)
            document = loader.load()
            for doc in document:
                doc.metadata.update({"source": file_path, "type": "pdf", "filename": Path(file_path).name})
            return document
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return []
        except Exception as e:
            logging.error(f"Error processing PDF file{file_path}: {str(e)}")
            return []

    @staticmethod
    def load_csv_file(file_path: str) -> List[Document]:
        """Load CSV file and convert to Documents"""
        try:
            df = pd.read_csv(file_path)
            csv_text = df.to_string(index=False)
            document = Document(page_content=csv_text,
                                metadata={"source": file_path, "type": "csv", "filename": Path(file_path).name, "columns": list(df.columns),
                                          "row_count": len(df)})
            return [document]
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return []
        except Exception as e:
            logging.error(f"Error processing CSV {file_path}: {str(e)}")
            return []


def load_documents(directory: str) -> List[Document]:
    """Load documents from directory"""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    documents: List[Document] = []
    processor = DocumentProcessor()

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1].lower()

            try:
                if file_extension == ".pdf":
                    document = processor.load_pdf_file(file_path)
                    if document:
                        documents.extend(document)
                elif file_extension in [".txt", ".doc", ".docx", "pptx"]:
                    document = processor.load_text_file(file_path)
                    if document:
                        documents.append(document)
                elif file_extension == ".csv":
                    document = processor.load_csv_file(file_path)
                    if documents:
                        documents.extend(document)
                elif file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                    document = processor.process_image(file_path)
                    if document:
                        documents.append(document)
                else:
                    logger.error(f"Unsupported file extension: {file_extension}")
            except FileNotFoundError:
                logger.error(f"File not found: {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
    if not documents:
        logger.error(f"No documents found in directory: {directory}")
    return documents


def create_vectorstore(documents: List[Document], vectorstore_type: str = "faiss") -> Union[FAISS, Chroma]:
    """Create vectorstore from documents"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    if vectorstore_type.lower() == "faiss":
        return FAISS.from_documents(texts, embeddings)
    else:
        return Chroma.from_documents(texts, embeddings)


def create_reranking_retriever(base_retriever, top_k=10):
    """Create a reranking retriever using Cohere"""
    compressor = CohereRerank(cohere_api_key=COHERE_API_KEY, top_n=top_k, model="rerank-multilingual-v2.0")
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)


def get_most_relevant_file(documents: List[Document], query: str) -> str:
    """Determine the most relevant file using semantic similarity and source type weights."""
    file_scores = {}
    type_weights = {"image": 1.2, "text": 1.0, "pdf": 1.0}
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    query_embedding = embeddings.embed_query(query)

    for doc in documents:
        filename = doc.metadata.get("filename", "unknown")
        doc_type = doc.metadata.get("type", "text")

        doc_embedding = embeddings.embed_query(doc.page_content)
        similarity = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
        weighted_score = similarity * type_weights.get(doc_type, 1.0)
        if filename not in file_scores:
            file_scores[filename] = {
                'score': weighted_score,
                'count': 1,
                'type': doc_type
            }
        else:
            current = file_scores[filename]
            current['score'] = (current['score'] * current['count'] + weighted_score) / (current['count'] + 1)
            current['count'] += 1
    if file_scores:
        return max(file_scores.items(), key=lambda x: x[1]['score'])[0]
    return "unknown"


def filter_documents_by_file(documents: List[Document], filename: str) -> List[Document]:
    """Filter documents to only include those from the specified file"""
    return [doc for doc in documents if doc.metadata.get("filename") == filename]


def filter_documents_by_relevance(documents: List[Document], query: str, top_k: int = 3) -> List[Document]:
    """Filter documents based on their relevance to the query."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    query_embedding = embeddings.embed_query(query)
    doc_scores = []
    for doc in documents:
        doc_embedding = embeddings.embed_query(doc.page_content)
        similarity = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
        doc_scores.append((doc, similarity))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in doc_scores[:top_k]]


def validate_context_relevance(source_documents: List[Document], question: str) -> bool:
    """Validate if the retrieved documents are actually relevant to the question"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    query_embedding = embeddings.embed_query(question)
    max_similarity = 0
    for doc in source_documents:
        doc_embedding = embeddings.embed_query(doc.page_content)
        similarity = np.dot(query_embedding, doc_embedding) / ( np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
        max_similarity = max(max_similarity, similarity)

    relevance_threshold = 0.5
    return max_similarity >= relevance_threshold


def initialize_rag(vectorstore_type: str = "faiss"):
    """Initialize the RAG system with improved context validation"""
    try:
        documents = load_documents(DOCS_DIR)
        logger.info(f"Total documents loaded: {len(documents)}")

        vectorstore = create_vectorstore(documents, vectorstore_type)
        logger.info(f"Vector store created using {vectorstore_type}")

        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
        reranking_retriever = create_reranking_retriever(base_retriever, top_k=5)

        prompt_template = """You are a knowledgeable expert who only provides information based on the given context. 
        If the question cannot be answered using the provided context, explicitly state that the information is not available 
        in the documents.

                Context: {context}

                Chat History: {chat_history}

                Question: {question}

                Instructions:
                1. First, evaluate if the context contains relevant information to answer the question
                2. If the context doesn't contain relevant information, respond with: "I cannot provide information about [topic] as it is not present in the available documents."
                3. If the context contains relevant information:
                   - Provide a clear answer using only information from the context
                   - Include specific details and examples from the context
                   - Organize the response in clear paragraphs
                4. Never make up or infer information that's not explicitly present in the context

                Response:"""
        custom_prompt = PromptTemplate(input_variables=["context", "chat_history", "question"], template=prompt_template)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=reranking_retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": custom_prompt},
            return_source_documents=True,
            return_generated_question=True
        )

        logger.info("RAG chain successfully initialized")
        return retrieval_chain

    except FileNotFoundError:
        logger.error(f"File not found: {DOCS_DIR}")
    except Exception as e:
        logger.error(f"Error processing RAG system: {traceback.format_exc()}")


def format_response(response: Dict, original_question: str) -> Dict:
    """Format the RAG response with improved relevance validation"""
    source_documents = response["source_documents"]

    is_relevant = validate_context_relevance(source_documents, original_question)

    if not is_relevant:
        topic = original_question.lower().replace("tell me about ", "").replace("what is ", "").strip()
        no_info_response = {
            "Question": original_question,
            "Answer": f"I cannot provide information about {topic} as it is not present in the available documents.",
            "Most Relevant File": "None",
            "Sources": [],
            "Generated Question": "None"
        }
        return no_info_response

    most_relevant_file = get_most_relevant_file(source_documents, original_question)
    relevant_docs = filter_documents_by_file(source_documents, most_relevant_file)
    filtered_documents = filter_documents_by_relevance(relevant_docs, original_question)

    return {
        "Question": response["question"],
        "Answer": response["answer"],
        "Most Relevant File": most_relevant_file,
        "Sources": [
            {
                "Source": f"Extract {i + 1}",
                "Content": doc.page_content[:200] + "...",
                "Type": doc.metadata.get("type", "unknown")
            }
            for i, doc in enumerate(filtered_documents)
        ],
        "Generated Question": response.get("generated_question", "Not available")
    }


def rag_response(question: str, vectorstore_type: str = "faiss") -> Dict:
    """Get and format response from the RAG system"""
    try:
        rag_chain = initialize_rag(vectorstore_type)
        response = rag_chain.invoke({"question": question})
        return format_response(response, question)
    except Exception as e:
        logger.error(f"Error getting RAG response: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        question = "tell me something about nepal "
        response = rag_response(question, vectorstore_type="faiss")
        print(response)

        print("\n=== RAG Response ===\n")
        print(f"Question: {response['Question']}\n")
        print(f"Answer: {response['Answer']}\n")
        print(f"Most Relevant File: {response['Most Relevant File']}\n")
        print("Relevant Extracts:")
        for source in response['Sources']:
            print(f"\n{source['Source']}:")
            print(f"{source['Content']}")
            print(f"Type: {source['Type']}")
        print(f"\nGenerated Question: {response['Generated Question']}")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
