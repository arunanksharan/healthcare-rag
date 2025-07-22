# RAG Demo: Document Retrieval Augmented Generation System

## Overview

This project is a Retrieval Augmented Generation (RAG) system designed for processing, indexing, and querying various types of official documents, such as government policies, guidelines, and FAQs. The system ingests documents (PDFs) through a FastAPI service. The ingestion process involves asynchronous task handling with Celery, document parsing and chunking using LlamaParse and custom logic, generating embeddings via multiple medical-specialized models, and storing the vectorized data in a locally hosted Qdrant database.

The retrieval component allows users to ask natural language questions. The query is enhanced, relevant document chunks are retrieved using vector search across multiple embedding spaces, re-ranked for improved relevance using a cross-encoder model, and finally, an LLM generates a synthesized answer with citations to the source documents.

The project is modularized into two main FastAPI applications:

- **Data Ingestion (`data_ingestion`)**: Handles receiving document ingestion requests, parsing documents, chunking them, generating embeddings, and indexing into Qdrant.
- **Data Retrieval (`data_retrieval`)**: Facilitates efficient querying, re-ranking, and LLM-based answer generation with citations.

## Features

- **FastAPI Endpoints**: Separate, robust APIs for document ingestion and data retrieval.
- **Local File Ingestion**: Accepts PDF files directly via a `multipart/form-data` endpoint, along with metadata (title, type, description, date, embedding_type).
- **Multiple Medical Embedding Models**: Supports PubMedBERT (default), BioBERT, SciBERT, ClinicalBERT, BioLinkBERT, and OpenAI embeddings for healthcare-specific document understanding.
- **Asynchronous Task Processing**: Leverages Celery (with Redis as the broker) for robust background processing of document ingestion tasks.
- **Advanced Document Parsing**: Uses LlamaParse for intelligent PDF parsing, extracting text, tables, and images along with their structural information.
- **Dual Chunking Strategies**: 
  - **Generic Chunker**: General-purpose chunking with FinBERT tokenizer (512 tokens, 50 overlap)
  - **Healthcare Chunker**: Medical-optimized chunking with PubMedBERT tokenizer (384 tokens, 64 overlap), section awareness, and medical content grouping
- **Multi-Collection Architecture**: Each embedding model has its own Qdrant collection to maintain vector space integrity.
- **Cross-Collection Search**: Can search across multiple embedding spaces simultaneously for comprehensive results.
- **Qdrant Vector Database**: Stores and indexes document chunks and their embeddings for efficient similarity search.
- **Query Enhancement**: Employs an LLM to refine user queries for better retrieval accuracy from official document types.
- **Cross-Encoder Reranking**: Uses a `sentence-transformers` cross-encoder model (e.g., `ms-marco-MiniLM-L-6-v2`) running locally to re-rank initial search results for improved relevance.
- **LLM-Powered Answer Generation**: Generates comprehensive answers to user queries based on retrieved and re-ranked contexts, complete with sequential citations to the source document chunks.
- **Detailed Citations**: Each citation includes the full metadata of the source chunk, providing rich contextual information.
- **Simplified Dependencies**: Removed S3 and Sentry integrations for a more streamlined local development experience.
- **Modular Design**: Clear separation between ingestion and retrieval services, promoting maintainability and scalability.

## Architecture

### Data Ingestion Flow

1.  **API Request**: User uploads a PDF file and associated metadata (title, type, description, date, embedding_type) to the `/ingest` endpoint in the `data_ingestion` service.
2.  **File Storage**: The uploaded file is temporarily saved to a local `media` directory.
3.  **Celery Task**: An asynchronous Celery task (`process_document_task`) is initiated.
4.  **Document Parsing**: The task uses LlamaParse (via `llama-cloud`) to parse the PDF file, extracting text, images, and structural elements.
5.  **Chunking**: The parsed content is processed by `CustomChunker`, which divides it into meaningful chunks (text, tables, images) while preserving page-level metadata.
6.  **Embedding**: Each chunk's content is embedded using the specified embedding model (defaults to PubMedBERT for healthcare).
7.  **Indexing**: The chunk content, its embedding, and comprehensive metadata are stored in an embedding-specific Qdrant collection.

### Data Retrieval Flow

1.  **API Request**: User sends a natural language `query` and optional `embedding_types` list to the `/search` endpoint in the `data_retrieval` service.
2.  **Query Enhancement**: The raw query is enhanced by an LLM (OpenAI GPT-4o) to be more specific and effective for retrieving information from official documents.
3.  **Embedding Generation**: The enhanced query is embedded using the specified embedding models (or all available models if not specified).
4.  **Vector Search**: The query embeddings are used to search the appropriate Qdrant collections for relevant document chunks.
5.  **Result Merging**: Results from multiple collections are combined and sorted by relevance score.
6.  **Reranking**: The top N candidate documents are re-ranked using a local cross-encoder model to improve the relevance ordering.
7.  **LLM Response Generation**: The re-ranked document chunks are passed as context to an LLM (OpenAI GPT-4.1) along with the enhanced query.
8.  **Synthesized Answer with Citations**: The LLM generates a bullet-point answer, citing the source contexts with full metadata.

## Embedding Models

The system supports multiple embedding models optimized for healthcare documents:

### Available Models

1. **PubMedBERT** (Default)
   - Model: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`
   - Dimension: 768
   - Pre-trained on 14M PubMed abstracts
   - Best for: Medical literature, clinical terms

2. **BioBERT**
   - Model: `dmis-lab/biobert-v1.1`
   - Dimension: 768
   - Pre-trained on biomedical corpus
   - Best for: Biomedical text

3. **SciBERT**
   - Model: `allenai/scibert_scivocab_uncased`
   - Dimension: 768
   - Pre-trained on scientific papers
   - Best for: Scientific literature

4. **ClinicalBERT**
   - Model: `emilyalsentzer/Bio_ClinicalBERT`
   - Dimension: 768
   - Pre-trained on clinical notes
   - Best for: Clinical documentation

5. **BioLinkBERT**
   - Model: `michiyasunaga/BioLinkBERT-base`
   - Dimension: 768
   - Optimized for biomedical entity linking
   - Best for: Entity-rich medical texts

6. **OpenAI**
   - Model: `text-embedding-ada-002`
   - Dimension: 1536
   - General-purpose embeddings
   - Best for: General text, administrative documents

### Downloading Models

Before first use, download all medical embedding models:

```bash
poetry run python scripts/download_models.py
```

This will cache all transformer models locally in the `model_cache` directory.

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
├── shared/               # Shared components between services
│   └── embeddings/       # Embedding models and registry
├── scripts/              # Utility scripts
│   └── download_models.py # Download embedding models
├── media/                # Temporary storage for uploaded files during ingestion
├── model_cache/          # Cached transformer models
├── README.md             # This file
├── pyproject.toml        # Project dependencies and metadata (Poetry)
├── poetry.lock           # Exact versions of dependencies
└── ... (other config files)
```

## Setup and Running

### Prerequisites

- Python 3.10+
- Poetry for dependency management
- Docker (and Docker Compose) for running Qdrant and Redis

### 1. Environment Variables

Create a `.env` file in the project root. Populate it with the following (replace placeholders with actual values):

```env
# OpenAI API Key
OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Qdrant Configuration
QDRANT_URL="http://localhost:6333"
QDRANT_COLLECTION_NAME="my_documents_collection" # Base name - embedding type will be appended

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

### 3. Download Embedding Models

```bash
poetry run python scripts/download_models.py
```

### 4. Start External Services (Qdrant & Redis)

A `docker-compose.yml` file is recommended for easily starting Qdrant and Redis. If you don't have one, create it:

```yaml
# docker-compose.yml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - '6333:6333'
      - '6334:6334'
    volumes:
      - ./qdrant_storage:/qdrant/storage # Persists Qdrant data
    restart: always

  redis:
    image: redis:latest
    ports:
      - '6379:6379'
    restart: always
```

Then run:

```bash
docker-compose up -d
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

Default ingestion (using PubMedBERT):

```bash
curl --location 'http://localhost:8000/ingestion/api/v1/ingest' \
--form 'file=@"/path/to/medical_document.pdf"' \
--form 'title="Medical Guidelines 2025"' \
--form 'description="Clinical practice guidelines"' \
--form 'type="GUIDELINE"' \
--form 'date="2025-01-15"'
```

Ingestion with specific embedding model:

```bash
curl --location 'http://localhost:8000/ingestion/api/v1/ingest' \
--form 'file=@"/path/to/medical_document.pdf"' \
--form 'title="Medical Guidelines 2025"' \
--form 'description="Clinical practice guidelines"' \
--form 'type="GUIDELINE"' \
--form 'date="2025-01-15"' \
--form 'embedding_type="biobert"'
```

Ingestion with specific chunker type:

```bash
curl --location 'http://localhost:8000/ingestion/api/v1/ingest' \
--form 'file=@"/path/to/medical_document.pdf"' \
--form 'title="Patient Clinical Notes"' \
--form 'description="Clinical documentation"' \
--form 'type="DOCUMENT"' \
--form 'date="2025-01-15"' \
--form 'chunker_type="healthcare"'  # Options: "generic" or "healthcare" (default)
```

#### Retrieval API Test

Search across all collections:

```bash
curl --location 'http://localhost:8001/retrieval/api/v1/search' \
--header 'Content-Type: application/json' \
--data '{
    "query": "What are the treatment guidelines for hypertension?"
}'
```

Search specific embedding types:

```bash
curl --location 'http://localhost:8001/retrieval/api/v1/search' \
--header 'Content-Type: application/json' \
--data '{
    "query": "What are the treatment guidelines for hypertension?",
    "embedding_types": ["pubmedbert", "clinicalbert"]
}'
```

## Key Technologies

- **FastAPI**: For building efficient and modern APIs.
- **Celery**: For distributed task queues.
- **Redis**: As the message broker and result backend for Celery.
- **Qdrant**: Vector database for storing and searching embeddings.
- **OpenAI API**: For generating text embeddings and LLM-powered responses.
- **LlamaParse (LlamaCloud)**: For advanced PDF document parsing.
- **Sentence Transformers**: For local cross-encoder reranking.
- **Transformers**: For medical BERT models (PubMedBERT, BioBERT, etc.).
- **PyTorch**: For running transformer models.
- **Poetry**: For dependency management.
- **Docker**: For containerizing and managing external services.

## Chunking Strategies

The system supports two chunking strategies:

### 1. Generic Chunker
- **Tokenizer**: FinBERT (financial domain)
- **Chunk Size**: 512 tokens
- **Overlap**: 50 tokens
- **Use Case**: General documents, non-medical content

### 2. Healthcare Chunker (Default)
- **Tokenizer**: PubMedBERT (medical domain)
- **Chunk Size**: 384 tokens (768 for tables)
- **Overlap**: 64 tokens
- **Features**:
  - Section-aware chunking (preserves medical document structure)
  - Medical section detection (Chief Complaint, Medications, Lab Results, etc.)
  - Smart content grouping (keeps medications, lab results together)
  - Context preservation (includes section headers in chunks)
  - Medical-specific chunk types (medication, lab_result, vital_signs)
- **Use Case**: Clinical notes, medical guidelines, healthcare documents

## Multi-Embedding Architecture

The system uses a multi-collection architecture where each embedding model has its own Qdrant collection. This design ensures:

1. **Vector Space Integrity**: Different embedding models produce incompatible vector spaces. Keeping them separate prevents meaningless comparisons.
2. **Flexible Search**: Can search specific collections or all collections based on use case.
3. **Model-Specific Optimization**: Each collection can be optimized for its specific embedding dimensions.
4. **Easy Model Addition**: New embedding models can be added without affecting existing data.

Collection naming convention: `{base_collection_name}_{embedding_type}`
Example: `my_documents_collection_pubmedbert`

## Future Enhancements (Potential)

- Support for more document types.
- Advanced metadata filtering options in retrieval.
- User authentication and authorization.
- Fine-tuning medical embeddings on domain-specific data.
- Hybrid search combining dense and sparse retrieval.
- Multi-modal embeddings for documents with images.
- Integration with medical knowledge graphs.
- UI for easier interaction.
