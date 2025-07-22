# RAG Demo: Document Retrieval Augmented Generation System

## Overview

This project is a Retrieval Augmented Generation (RAG) system designed for processing, indexing, and querying various types of official documents, such as government policies, guidelines, and FAQs. The system ingests documents (PDFs) through a FastAPI service. The ingestion process involves asynchronous task handling with Celery, document parsing and chunking using LlamaParse and custom logic, generating embeddings via OpenAI, and storing the vectorized data in a locally hosted Qdrant database.

The retrieval component allows users to ask natural language questions. The query is enhanced, relevant document chunks are retrieved using vector search, re-ranked for improved relevance using a cross-encoder model, and finally, an LLM generates a synthesized answer with citations to the source documents.

The project is modularized into two main FastAPI applications:

-   **Data Ingestion (`data_ingestion`)**: Handles receiving document ingestion requests, parsing documents, chunking them, generating embeddings, and indexing into Qdrant.
-   **Data Retrieval (`data_retrieval`)**: Facilitates efficient querying, re-ranking, and LLM-based answer generation with citations.

## Features

-   **FastAPI Endpoints**: Separate, robust APIs for document ingestion and data retrieval.
-   **Local File Ingestion**: Accepts PDF files directly via a `multipart/form-data` endpoint, along with metadata (title, type, description, date).
-   **Asynchronous Task Processing**: Leverages Celery (with Redis as the broker) for robust background processing of document ingestion tasks.
-   **Advanced Document Parsing**: Uses LlamaParse for intelligent PDF parsing, extracting text, tables, and images along with their structural information.
-   **Custom Chunking**: Implements sophisticated chunking strategies that respect document structure (headings, paragraphs, tables, images) and includes page-level metadata like `page_width` and `page_height`.
-   **OpenAI Embeddings**: Utilizes OpenAI's text embedding models to generate vector representations of document chunks.
-   **Qdrant Vector Database**: Stores and indexes document chunks and their embeddings for efficient similarity search.
-   **Query Enhancement**: Employs an LLM to refine user queries for better retrieval accuracy from official document types.
-   **Cross-Encoder Reranking**: Uses a `sentence-transformers` cross-encoder model (e.g., `ms-marco-MiniLM-L-6-v2`) running locally to re-rank initial search results for improved relevance.
-   **LLM-Powered Answer Generation**: Generates comprehensive answers to user queries based on retrieved and re-ranked contexts, complete with sequential citations to the source document chunks.
-   **Detailed Citations**: Each citation includes the full metadata of the source chunk, providing rich contextual information.
-   **Simplified Dependencies**: Removed S3 and Sentry integrations for a more streamlined local development experience.
-   **Modular Design**: Clear separation between ingestion and retrieval services, promoting maintainability and scalability.

## Architecture

### Data Ingestion Flow

1.  **API Request**: User uploads a PDF file and associated metadata (title, type, description, date) to the `/ingest` endpoint in the `data_ingestion` service.
2.  **File Storage**: The uploaded file is temporarily saved to a local `media` directory.
3.  **Celery Task**: An asynchronous Celery task (`process_document_task`) is initiated.
4.  **Document Parsing**: The task uses LlamaParse (via `llama-cloud`) to parse the PDF file, extracting text, images, and structural elements.
5.  **Chunking**: The parsed content is processed by `CustomChunker`, which divides it into meaningful chunks (text, tables, images) while preserving page-level metadata.
6.  **Embedding**: Each chunk's content is embedded using OpenAI's API.
7.  **Indexing**: The chunk content, its embedding, and comprehensive metadata (including `original_filename`, `title`, `type`, `description`, `date`, `page_number`, `page_width`, `page_height`, `parse_type`) are stored in a Qdrant collection.

### Data Retrieval Flow

1.  **API Request**: User sends a natural language `query` to the `/search` endpoint in the `data_retrieval` service.
2.  **Query Enhancement**: The raw query is enhanced by an LLM (OpenAI GPT-4o) to be more specific and effective for retrieving information from official documents.
3.  **Embedding Generation**: The enhanced query is embedded using OpenAI's API.
4.  **Vector Search**: The query embedding is used to search the Qdrant collection for relevant document chunks. This search is performed without metadata filters by default.
5.  **Reranking**: The top N candidate documents from Qdrant are re-ranked using a local cross-encoder model to improve the relevance ordering.
6.  **LLM Response Generation**: The re-ranked document chunks are passed as context to an LLM (OpenAI GPT-4.1) along with the enhanced query.
7.  **Synthesized Answer with Citations**: The LLM generates a bullet-point answer, citing the source contexts (e.g., `^[1]`, `^[2]`). Each citation in the response includes the full metadata of the original document chunk.

## Project Structure

```
rag_demo/
├── .env                  # Local environment variables (see Configuration)
├── data_ingestion/       # Data Ingestion FastAPI service
│   ├── app/              # API routes, models
│   ├── core/             # Core settings, main app setup
│   ├── tasks/            # Celery tasks for document processing
│   └── utils/            # Parsers, chunkers, helpers
├── data_retrieval/       # Data Retrieval FastAPI service
│   ├── app/              # API routes, models
│   ├── core/             # Core settings, main app setup
│   └── utils/            # Search, reranker, LLM generator, prompts
├── media/                # Temporary storage for uploaded files during ingestion
├── README.md             # This file
├── pyproject.toml        # Project dependencies and metadata (Poetry)
├── poetry.lock           # Exact versions of dependencies
└── ... (other config files)
```

## Setup and Running

### Prerequisites

-   Python 3.10+
-   Poetry for dependency management
-   Docker (and Docker Compose) for running Qdrant and Redis

### 1. Environment Variables

Create a `.env` file in the project root. Populate it with the following (replace placeholders with actual values):

```env
# OpenAI API Key
OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Qdrant Configuration
QDRANT_URL="http://localhost:6333"
QDRANT_COLLECTION_NAME="my_documents_collection_llama_parse_v3" # Or your chosen collection name

# LlamaParse (LlamaCloud) API Key
LLAMA_CLOUD_API_KEY="llx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Celery Configuration (using Redis as broker and backend)
CELERY_BROKER_URL="redis://localhost:6379/0"
CELERY_RESULT_BACKEND="redis://localhost:6379/0"

# Optional: Logging Level (e.g., INFO, DEBUG, WARNING)
LOG_LEVEL="INFO"
```

### 2. Install Dependencies

```bash
poetry install
```

### 3. Start External Services (Qdrant & Redis)

A `docker-compose.yml` file is recommended for easily starting Qdrant and Redis. If you don't have one, create it:

```yaml
# docker-compose.yml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage # Persists Qdrant data
    restart: always

  redis:
    image: redis:latest
    ports:
      - "6379:6379"
    restart: always
```

Then run:
```bash
docker-compose up -d
```

### 4. Run Database Migrations (if applicable for Qdrant setup)

Qdrant collections are typically created on-the-fly by the application when data is first inserted if they don't exist. Ensure your `QDRANT_COLLECTION_NAME` in `.env` matches what the application expects (see `data_ingestion/core/settings.py` or similar).

## Key Technologies

-   **FastAPI**: For building efficient and modern APIs.
-   **Celery**: For distributed task queues.
-   **Redis**: As the message broker and result backend for Celery.
-   **Qdrant**: Vector database for storing and searching embeddings.
-   **OpenAI API**: For generating text embeddings and LLM-powered responses.
-   **LlamaParse (LlamaCloud)**: For advanced PDF document parsing.
-   **Sentence Transformers**: For local cross-encoder reranking.
-   **Poetry**: For dependency management.
-   **Docker**: For containerizing and managing external services.

## Future Enhancements (Potential)

-   Support for more document types.
-   Advanced metadata filtering options in retrieval.
-   User authentication and authorization.
-   More sophisticated query analysis and decomposition.
-   Integration with knowledge graphs.
-   UI for easier interaction.
```

```dotenv
QDRANT_URL=
QDRANT_COLLECTION_NAME=
OPENAI_API_KEY=
REDIS_URL=
LLAMA_CLOUD_API_KEY=
PYTHONOPTIMIZE=1
CELERY_MP_START_METHOD=spawn
```

## Usage

### Running Services

#### Celery Worker

```bash
poetry run celery -A data_ingestion.celery_worker.tasks worker --loglevel=info --pool=prefork -Ofair --concurrency=4
```

#### Ingestion Service

```bash
poetry run uvicorn data_ingestion.core.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Retrieval Service

```bash
poetry run uvicorn data_retrieval.core.main:app --host 0.0.0.0 --port 8001 --reload
```

### API Testing

#### Ingestion API Test

```bash
curl --location 'http://localhost:8000/ingestion/api/v1/ingest' \
--form 'file=@"/Users/akshat_g/Downloads/Archive/93f32e63-eb70-472c-8e06-5f426db30864_Institutions_involved.pdf"' \
--form 'title="policy_file_2025"' \
--form 'description="File contains the policies of 2025"' \
--form 'type="POLICY"' \
--form 'date="2024-06-15"'
```

#### Retrieval API Test

```bash
curl --location 'http://localhost:8001/retrieval/api/v1/search' \
--header 'Content-Type: application/json' \
--data '{
    "query": "What is NDDB?"
}'
```
