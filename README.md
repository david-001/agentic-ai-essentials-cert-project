# RAG-Based AI Assistant for Document Q&A

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.4.1-orange.svg)](https://www.trychroma.com/)

## 📑 Table of Contents

- [Project Summary](#-project-summary)
- [Project Overview](#-project-overview)
  - [What This Project Does](#what-this-project-does)
  - [How It Works](#how-it-works)
  - [Technical Approach](#technical-approach)
- [Key Features](#-key-features)
  - [Document Processing](#document-processing)
  - [Search & Retrieval](#search--retrieval)
  - [LLM Integration](#llm-integration)
  - [Configuration & Flexibility](#configuration--flexibility)
  - [Testing & Evaluation](#testing--evaluation)
- [Repository Structure](#-repository-structure)
  - [Key Files Explained](#key-files-explained)
- [Code Architecture](#️-code-architecture)
  - [Core Components](#core-components)
  - [Modular Test Architecture](#modular-test-architecture)
- [Installation](#-installation)
  - [Prerequisites](#prerequisites)
  - [Step-by-Step Setup](#step-by-step-setup)
  - [Verify Installation](#verify-installation)
- [Usage](#-usage)
  - [Basic Usage](#basic-usage)
  - [Example Session](#example-session)
  - [Example Questions](#example-questions)
- [Configuration](#️-configuration)
  - [YAML Configuration](#yaml-configuration)
  - [Environment Variables](#environment-variables)
  - [Supported LLM Providers](#supported-llm-providers)
- [Testing](#-testing)
  - [Test Categories](#test-categories)
  - [Running Tests](#running-tests)
  - [RAG Evaluation Metrics](#rag-evaluation-metrics)
- [Troubleshooting](#-troubleshooting)
  - [Common Issues and Solutions](#common-issues-and-solutions)
  - [Known Limitations](#known-limitations)
- [Additional Resources](#-additional-resources)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Support & Contact](#-support--contact)
  - [Getting Help](#getting-help)
  - [Contributing](#contributing)

---

## 📋 Project Summary

An intelligent question-answering system that uses **Retrieval-Augmented Generation (RAG)** to provide accurate answers about the **TaskFlow Pro** project management platform. Built as part of the [Ready Tensor Agentic AI Essentials certification program](https://www.readytensor.ai/agentic-ai-essentials-cert/), this assistant combines semantic search with large language models and a two-stage query optimization pipeline to create a domain-aware knowledge base that you can query in natural language.

**Key Capabilities:**
- 🔍 Semantic document search using vector embeddings
- 🧠 LLM-based query optimization — rewrites user queries before retrieval for better results
- 💬 Natural language question answering with source attribution
- 📚 Support for multiple document formats (.txt, .md)
- 🏷️ Domain-aware document metadata (type, description, search keywords)
- 🤖 Multi-provider LLM support (OpenAI, Groq, Google Gemini)
- 💾 Persistent vector storage with ChromaDB
- ⚙️ YAML-based configuration for easy customization
- 🔄 Intelligent text chunking with RecursiveCharacterTextSplitter
- 🧪 Comprehensive test suite with DeepEval quality metrics
- 📊 Advanced RAG evaluation (Precision, Recall, MRR, NDCG, Faithfulness, Relevancy)

---

## 🎯 Project Overview

### What This Project Does

This RAG assistant is purpose-built for the **TaskFlow Pro** project management platform knowledge base. Unlike general-purpose chatbots that only know information from their training data, this system:

1. **Ingests** TaskFlow Pro documents (API docs, company policies, customer FAQ, product docs, security & compliance)
2. **Indexes** them using semantic embeddings with domain-aware metadata for intelligent search
3. **Optimizes** your query — an LLM rewrites and expands it before retrieval
4. **Retrieves** the most relevant information when you ask a question
5. **Generates** accurate, context-aware answers with source attribution using state-of-the-art LLMs

### How It Works

The system implements a **two-stage RAG pipeline** with LLM query optimization:

```
┌─────────────────┐
│  Your Documents │  (TaskFlow Pro: API docs, policies, FAQ, etc.)
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│ Text Chunking   │────▶│ Create Embeddings│
│ (512 chars,     │     │ (Vector Space)   │
│  50 overlap)    │     │                  │
└─────────────────┘     └────────┬─────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   Store in      │
                        │   ChromaDB      │
                        │ (with domain    │
                        │  metadata)      │
                        └────────┬────────┘
                                 │
         ┌───────────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  User Question  │────▶│  Stage 1: LLM    │────▶│  Stage 2:        │
│                 │     │  Query           │     │  Semantic Search │
└─────────────────┘     │  Optimization    │     │  (Find Top N)    │
                        │  (Rewrite &      │     └────────┬─────────┘
                        │  Expand Query)   │              │
                        └──────────────────┘              ▼
                                                ┌──────────────────┐
                                                │  LLM Generates   │
                                                │  Answer with     │
                                                │  Source Citation │
                                                └──────────────────┘
```

**Step-by-Step Process:**

1. **Document Loading**: Reads `.txt` and `.md` files from the `data/` directory with domain-specific metadata (document type, description, search keywords)
2. **Text Chunking**: Uses LangChain's RecursiveCharacterTextSplitter (512-char chunks, 50-char overlap)
3. **Embedding Creation**: Converts text chunks to 384-dimensional vectors using sentence-transformers
4. **Vector Storage**: Stores embeddings in ChromaDB with persistent local storage
5. **Query Processing**: When you ask a question:
   - **Stage 1 — Query Optimization**: The LLM rewrites your raw question into an optimized retrieval query (expands abbreviations, adds domain synonyms, removes filler words)
   - **Stage 2 — Retrieval & Generation**: The optimized query finds the most similar document chunks; the LLM generates an answer with source attribution from the retrieved context

### Technical Approach

**Two-Stage RAG Pipeline:**
The core innovation in this project is the two-stage pipeline. Rather than searching with the raw user question, the system first sends it through an LLM-based **query optimizer** that expands abbreviations (e.g., "2FA" → "two-factor authentication"), adds domain synonyms likely to appear in TaskFlow Pro documents, and strips filler words. Only then does it perform the semantic search. This consistently improves retrieval precision.

**Vector Search vs. Keyword Search:**
Traditional keyword search looks for exact word matches. This RAG system uses *semantic search*, which understands meaning. For example:
- Question: "How many days off do I get?"
- Matches documents about: "vacation policy", "paid time off", "leave entitlement"
- Even if the exact words "days off" don't appear

**Why RAG Instead of Fine-Tuning:**
- ✅ No expensive model retraining needed
- ✅ Update knowledge base by just adding documents
- ✅ Reduces hallucinations (answers based on your docs)
- ✅ Transparent and explainable results with source attribution

---

## 🌟 Key Features

### Document Processing
- 📚 **Automatic Loading**: Scans `data/` folder for `.txt` and `.md` files
- 🏷️ **Domain Metadata**: Each document is tagged with its type (API Docs, FAQ, Policies, etc.), description, and search keywords
- 🔄 **Smart Chunking**: RecursiveCharacterTextSplitter with 512-char chunks and 50-char overlap preserves context across chunks
- 💾 **Persistent Storage**: ChromaDB with local file-based persistence

### Search & Retrieval
- 🧠 **Query Optimization**: LLM rewrites user queries for better retrieval — expands abbreviations, adds domain synonyms, removes filler
- 🔍 **Semantic Search**: sentence-transformers/all-MiniLM-L6-v2 (384-dim embeddings)
- 📊 **Configurable Results**: Retrieve top N most relevant chunks (default: 3)
- 🎯 **Source Attribution**: Answers cite the document type and filename they came from

### LLM Integration
- 🤖 **Multi-Provider Support**: 
  - OpenAI (GPT-4o-mini)
  - Groq (Llama-3.1-8b-instant)
  - Google (Gemini-2.0-flash)

### Configuration & Flexibility
- ⚙️ **YAML Configuration**: Edit `config/config.yaml` for easy customization
- 🔐 **Secure API Keys**: Environment-based secrets via `.env` file
- 📝 **Model Selection**: Choose LLM provider and model via environment variables

### Testing & Evaluation
- 🧪 **Comprehensive Tests**: pytest-based unit and integration tests
- 📈 **DeepEval Integration**: Advanced quality metrics
- 🎯 **Retrieval Metrics**: Precision, Recall, MRR, NDCG
- 🔍 **Generation Metrics**: Faithfulness, Answer Relevancy, Contextual Precision/Recall
- 🎲 **Reproducible Tests**: Explicit random seed setting for consistent results
- 🏗️ **Modular Architecture**: Separated utility modules for maintainability

---

## 📁 Repository Structure

```
agentic-ai-essentials-cert-project/
│
├── src/                          # Source code directory
│   ├── app.py                    # Main application with RAG pipeline
│   ├── config.py                 # Configuration loader (loads from YAML)
│   └── vectordb.py               # Vector database wrapper for ChromaDB
│
├── config/                       # Configuration directory
│   └── config.yaml               # YAML configuration file (edit settings here)
│
├── data/                         # Document collection
│   ├── api_documentation.md      # Sample: API documentation
│   ├── company_policies.md       # Sample: HR policies
│   ├── customer_faq.md           # Sample: Customer FAQ
│   ├── product_documentation.md  # Sample: Product information
│   └── security_compliance.md    # Sample: Security documentation
│
├── tests/                        # Comprehensive test suite
│   ├── conftest.py               # Pytest configuration and shared fixtures
│   ├── metrics_utils.py          # Metric calculation utilities
│   ├── rag_evaluator.py          # DeepEval-based RAG quality evaluator
│   ├── rag_evaluator_utils.py    # Helper utilities for evaluation
│   ├── test_app.py               # Integration tests for RAG pipeline
│   └── test_vectordb.py          # Unit tests for vector database
│
├── requirements.txt              # Python dependencies
├── pytest.ini                    # Pytest configuration
├── .env                          # Environment variables (API keys) - DO NOT COMMIT
├── .env.example                  # Template for environment setup
├── .gitignore                    # Git ignore rules
├── LICENSE                       # MIT License
└── README.md                     # This file
│
└── chroma_db/                    # Vector database storage (auto-created)

```

### Key Files Explained

| File/Directory | Purpose | When to Modify |
|----------------|---------|----------------|
| `src/app.py` | Main RAG application entry point | Add features, modify prompt template |
| `src/vectordb.py` | ChromaDB wrapper with chunking & embedding | Adjust chunk size, embedding model |
| `src/config.py` | Loads configuration from YAML | Rarely (auto-loads from config.yaml) |
| `config/config.yaml` | Main configuration file | Change settings here |
| `.env` | API keys and secrets | Set your API keys here |
| `.env.example` | Template for environment variables | Reference for setup |
| `data/` | Your document collection | Add your .txt or .md files |
| `tests/conftest.py` | Pytest fixtures and configuration | Add new fixtures or test markers |
| `tests/rag_evaluator.py` | RAG system quality evaluator | Add new evaluation metric |
| `tests/metrics_utils.py` | RAG system quality evaluator utilies | Add new helper functions to RAG system quality evaluator |
| `tests/test_app.py` | Integration tests for RAG pipeline | Add new integration tests |
| `tests/test_vectordb.py` | Unit tests for vector database | Add new unit tests |
| `pytest.ini` | Test runner configuration | Modify test settings |

---

## 🏗️ Code Architecture

### Core Components

**1. Document Loader (`app.py:load_documents`)**
- Scans `data/` directory for supported file types (`.txt`, `.md`)
- Attaches domain-specific metadata to each document: `document_type`, `document_description`, `search_keywords`, `domain`
- Uses `DOCUMENT_TYPE_MAP` to classify documents (customer_faq, api_documentation, company_policies, product_documentation, security_compliance)

**2. VectorDB (`vectordb.py:VectorDB`)**
- Wraps ChromaDB for vector storage
- Uses sentence-transformers for embeddings (384-dim, all-MiniLM-L6-v2)
- Implements intelligent text chunking (512-char chunks, 50-char overlap)
- Provides semantic search functionality returning documents, metadatas, distances, and IDs

**3. RAG Assistant (`app.py:RAGAssistant`)**
- Orchestrates the complete two-stage RAG pipeline
- **Stage 1 — Query Optimization**: `_optimize_query()` uses the LLM to rewrite user queries for better retrieval
- **Stage 2 — Retrieval & Generation**: `query()` retrieves top-N chunks and generates answers with `_format_context()` providing source attribution
- Supports multiple LLM providers (OpenAI, Groq, Google)
- Uses LangChain for prompt templating; constrains answers strictly to provided context

**4. Configuration (`config.py`)**
- Loads settings from YAML
- Exports as Python constants
- Supports environment variable overrides

### Modular Test Architecture

**Separated Concerns:**
- `conftest.py`: Shared pytest fixtures and configuration
- `metrics_utils.py`: Isolated metric calculation functions
- `rag_evaluator.py`: High-level evaluation orchestration
- `rag_evaluator_utils.py`: Helper utilities for evaluation
- `test_app.py`: Integration tests
- `test_vectordb.py`: Unit tests

**Key Design Patterns:**
- **Fixtures**: Reusable test data and mock objects
- **Markers**: Categorize tests (`@pytest.mark.unit`, `@pytest.mark.integration`)
- **Utilities**: Pure functions for metrics (easy to test and reuse)
- **Separation**: Clear boundaries between test types

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.10 or higher (3.11 recommended)
- **pip**: Python package manager (included with Python)
- **API Key**: At least one is required:
  - [OpenAI API key](https://platform.openai.com/api-keys)
  - [Groq API key](https://console.groq.com/keys)
  - [Google Gemini API key](https://aistudio.google.com/api-keys/)

### Step-by-Step Setup

**1. Clone or Download the Repository**
```bash
git clone https://github.com/david-001/agentic-ai-essentials-cert-project.git
cd agentic-ai-essentials-cert-project
```

**2. Create a Virtual Environment** (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

This installs all required packages including:

**Core RAG Components:**
- `chromadb==1.4.1` - Vector database for embeddings storage
- `sentence-transformers==5.2.0` - Embedding models for semantic search
- `langchain-core==1.2.7` - Core LangChain functionality
- `langchain-text-splitters==1.1.0` - Text chunking utilities

**LLM Integrations:**
- `langchain-openai==1.1.7` - OpenAI GPT integration
- `langchain-groq==1.1.1` - Groq LLM integration
- `langchain-google-genai==4.2.0` - Google Gemini integration
- `openai==2.15.0` - OpenAI API client
- `groq==0.37.1` - Groq API client
- `google-genai==1.59.0` - Google Generative AI client

**Testing & Evaluation:**
- `pytest==9.0.2` - Testing framework
- `pytest-asyncio==1.3.0` - Async test support
- `pytest-repeat==0.9.4` - Test repetition
- `pytest-rerunfailures==16.1` - Auto-retry failed tests
- `pytest-xdist==3.8.0` - Parallel test execution
- `deepeval==3.8.0` - RAG evaluation metrics (Faithfulness, Relevancy, etc.)

**Machine Learning & Data Processing:**
- `torch==2.9.1` - PyTorch deep learning framework
- `transformers==4.57.6` - Hugging Face transformers
- `numpy==2.4.1` - Numerical computing
- `scikit-learn==1.8.0` - Machine learning utilities
- `scipy==1.17.0` - Scientific computing

**Configuration & Utilities:**
- `python-dotenv==1.2.1` - Environment variable management
- `PyYAML==6.0.3` - YAML configuration parsing
- `pydantic==2.12.5` - Data validation
- `pydantic-settings==2.12.0` - Settings management

**API & Networking:**
- `httpx==0.28.1` - Modern HTTP client
- `aiohttp==3.13.3` - Async HTTP client
- `requests==2.32.5` - HTTP library

**Additional Dependencies:**
- All other packages listed in `requirements.txt` including CUDA support for GPU acceleration, monitoring tools, and various utilities

**4. Configure Environment Variables**
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API key
# Use your preferred text editor (nano, vim, code, etc.)
nano .env
```

**Example `.env` file:**
```bash
# Choose ONE provider (or multiple for fallback)

# Option 1: OpenAI (Recommended for quality)
OPENAI_API_KEY=sk-proj-your-actual-key-here
OPENAI_MODEL=gpt-4o-mini

# Option 2: Groq (Recommended for speed)
GROQ_API_KEY=gsk-your-actual-key-here
GROQ_MODEL=llama-3.1-8b-instant

# Option 3: Google Gemini (Generous free tier)
GOOGLE_API_KEY=AIza-your-actual-key-here
GOOGLE_MODEL=gemini-2.0-flash
```

**Important:** Remove the placeholder text and replace with your actual API key!

**5. Add Your Documents** (Optional)
```bash
# The project includes 5 sample documents (.md format):
data/
  ├── api_documentation.md
  ├── company_policies.md
  ├── customer_faq.md
  ├── product_documentation.md
  └── security_compliance.md

# You can add your own .txt or .md files:
cp /path/to/your/document.txt data/
cp /path/to/your/document.md data/
```

### Verify Installation

```bash
# Run tests to verify everything works
pytest

# Or run with verbose output
pytest -v

# If you see tests passing, you're good to go!
```

---

## 💡 Usage

### Basic Usage

**Start the Interactive Assistant:**
```bash
python src/app.py
```

**What Happens:**
1. System initializes LLM (checks for API keys in order: OpenAI → Groq → Google)
2. Loads documents from `data/` directory
3. Chunks documents and creates embeddings
4. Stores vectors in ChromaDB
5. Starts interactive Q&A session

### Example Session

```bash
$ python src/app.py

Initializing RAG Assistant...
Using Google Gemini model: gemini-2.0-flash
Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
Vector database initialized with collection: rag_documents
RAG Assistant initialized successfully

Loading documents...
Loaded [API Documentation]: api_documentation.md
Loaded [Customer FAQ]: customer_faq.md
Loaded [Company Policies]: company_policies.md
Loaded [Security & Compliance]: security_compliance.md
Loaded [Product Documentation]: product_documentation.md
Loaded 5 documents
Processing 5 documents...
Document 1: Split into 33 chunks
Document 2: Split into 49 chunks
Document 3: Split into 36 chunks
Document 4: Split into 44 chunks
Document 5: Split into 40 chunks
Creating embeddings for 202 chunks...
Batches: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:02<00:00,  3.34it/s]
Adding to vector database...

Enter a question or 'quit' to exit: What vacation days do employees get?
  [Query Optimization] Original: 'What vacation days do employees get?'
  [Query Optimization] Optimized: 'Employee vacation days policy; employee time off benefits; TaskFlow Pro company policies.'

According to the Company Policies:

**Vacation Days:**
*   0-2 years: 20 days per year
*   3-5 years: 25 days per year
*   5+ years: 30 days per year
*   Accrued monthly and available immediately
*   Unused days roll over up to a maximum of 10 days
```

### Example Questions

Try these questions with the sample documents:

**Company Policies:**
- "What is the remote work policy?"
- "How many vacation days do employees get?"
- "Does the company provide health insurance?"

**API Documentation:**
- "What's the API rate limit for the Professional plan?"
- "How do I authenticate with the API?"
- "How do webhooks work?"

**Product Information:**
- "What pricing plans are available?"
- "What integrations does the product support?"
- "Is there a free trial?"

**Security & Compliance:**
- "Is the platform GDPR compliant?"
- "What encryption is used?"
- "What is the data retention policy?"

---

## ⚙️ Configuration

### YAML Configuration

Edit `config/config.yaml` to customize system behavior:

```yaml
# RAG Assistant - Configuration File

# ============================================================================
# Embedding Model Configuration
# ============================================================================
embedding:
  model: sentence-transformers/all-MiniLM-L6-v2

# ============================================================================
# Vector Database Configuration
# ============================================================================
database:
  collection_name: rag_documents  # ChromaDB collection name
  path: ./chroma_db               # Path to store vector database

# ============================================================================
# LLM Configuration
# ============================================================================
llm:
  temperature: 0.0         # 0.0 = deterministic, 1.0 = creative
  # Lower temperature = more consistent answers
  # Higher temperature = more varied/creative answers

# ============================================================================
# File Paths
# ============================================================================
paths:
  data_directory: data     # Directory containing your documents

# ============================================================================
# Chunking Configuration
# ============================================================================
chunking:
  chunk_size: 512        # Characters per chunk (optimised for TaskFlow Pro markdown docs)
  chunk_overlap: 50      # Overlap to preserve context across chunk boundaries

# ============================================================================
# RAG Query Configuration
# ============================================================================
rag:
  default_n_results: 3      # Number of chunks to retrieve per query
  query_optimization: true  # Enable LLM-based query rewriting before retrieval
```

**After Changing Configuration:**
```bash
# No restart needed - config is loaded on each run
python src/app.py

# For tests, pytest automatically picks up changes
pytest
```

### Environment Variables

Configure in `.env` file:

```bash
# ============================================================================
# LLM Provider Selection (choose one or multiple for fallback)
# ============================================================================

# OpenAI Configuration
OPENAI_API_KEY=sk-proj-your-key-here
OPENAI_MODEL=gpt-4o-mini          # or gpt-4o, gpt-3.5-turbo

# Groq Configuration
GROQ_API_KEY=gsk-your-key-here
GROQ_MODEL=llama-3.1-8b-instant   # or llama-3.1-70b-versatile

# Google Gemini Configuration
GOOGLE_API_KEY=AIza-your-key-here
GOOGLE_MODEL=gemini-2.0-flash     # or gemini-2.0-flash-exp

```

### Supported LLM Providers

The system checks for API keys in this order:
1. **OpenAI** (best quality, moderate cost)
2. **Groq** (fastest inference, free tier)
3. **Google Gemini** (generous free tier, good quality)

**Provider Comparison:**

| Provider | Speed | Quality | Free Tier | Best For |
|----------|-------|---------|-----------|----------|
| OpenAI GPT-4o-mini | Fast | Excellent | Limited | Production use |
| Groq Llama-3.1 | Very Fast | Good | Generous | Development/testing |
| Google Gemini-2.0 | Fast | Very Good | Very Generous | Cost-conscious projects |

---

## 🧪 Testing

### Test Categories

Tests are organized with pytest markers:

- **Unit Tests** (`@pytest.mark.unit`): Test individual components in isolation
  - Vector database operations
  - Embedding creation
  - Text chunking
  
- **Integration Tests** (`@pytest.mark.integration`): Test complete workflows
  - End-to-end RAG pipeline
  - Document loading and retrieval
  - Answer generation

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run specific test file
pytest tests/test_vectordb.py

# Run specific test function
pytest tests/test_app.py::TestRAGPipeline::test_add_documents

```

### RAG Evaluation Metrics

The project includes comprehensive RAG evaluation using DeepEval:

**Retrieval Metrics:**
- **Precision@K**: Proportion of retrieved documents that are relevant
- **Recall@K**: Proportion of relevant documents that are retrieved
- **MRR (Mean Reciprocal Rank)**: How early the first relevant result appears
- **NDCG (Normalized Discounted Cumulative Gain)**: Quality of ranking

**Generation Metrics (DeepEval):**
- **Faithfulness**: Answer is grounded in retrieved context (no hallucination)
- **Answer Relevancy**: Answer addresses the question
- **Contextual Precision**: Retrieved context is relevant to the question
- **Contextual Recall**: All necessary context is retrieved
- **Contextual Relevancy**: Overall context quality

**Run Evaluation:**
```bash
# Run evaluation
python tests/rag_evaluator.py

```

**Sample Output:**
```
======================================================================
RAG SYSTEM PERFORMANCE REPORT (DeepEval)
======================================================================

Generated: 2026-02-28T06:20:27.811803
Model: <deepeval.models.llms.gemini_model.GeminiModel object at 0x7fcdd09d4b60>

----------------------------------------------------------------------
RETRIEVAL METRICS:
  Precision@3:  0.3152
  Recall@3:     0.9455
  MRR:          0.8804
  NDCG@5:       0.8972
  Avg Latency:  22.94ms

GENERATION METRICS (DeepEval):
  Faithfulness:          0.9971
  Answer Relevance:      0.9481
  Contextual Precision:  0.9138
  Contextual Recall:     0.9513
  Contextual Relevancy:  0.4505

======================================================================
OVERALL GRADE: A (Excellent)
======================================================================

======================================================================
Performance evaluation complete!
======================================================================
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### "Environment variable not set" or "No valid API key found"

**Solution:**
```bash
# 1. Create .env file if it doesn't exist
cp .env.example .env

# 2. Edit .env and add your API key
nano .env

# 3. Add at least one API key (remove placeholder text):
OPENAI_API_KEY=sk-proj-YOUR-ACTUAL-KEY-HERE

# 4. Ensure no spaces around the = sign
# 5. Never commit .env to version control
```

#### "ModuleNotFoundError: No module named 'X'"

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstall all dependencies
pip install -r requirements.txt

# If still failing, try force reinstall
pip install --force-reinstall -r requirements.txt

# Verify Python version (should be 3.10+)
python --version
```

#### "No documents found in data directory"

**Solution:**
```bash
# Check data directory exists
ls -la data/

# Add .txt or .md files
cp your_document.txt data/

# Verify files are readable
cat data/your_document.txt

# Ensure files have content (not empty)
wc -l data/*.txt data/*.md
```

#### "Rate limit exceeded"

**Solution:**
- **Switch providers**: Use Google Gemini (most generous free tier)
  ```bash
  # In .env, comment out OpenAI and use:
  GOOGLE_API_KEY=AIza-your-key-here
  GOOGLE_MODEL=gemini-2.0-flash
  ```
- **Wait and retry**: Rate limits reset after time period
- **Upgrade plan**: Consider paid API tier for higher limits

#### "Python version not supported"

**Solution:**
```bash
# Check your Python version
python --version

# Should be Python 3.10 or higher
# If not, install Python 3.10+ and create new virtual environment:
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### "ImportError: cannot import name 'config'"

**Solution:**
```bash
# Ensure you're running from the project root
cd agentic-ai-essentials-cert-project

# Run with proper Python path
python src/app.py

```

#### "Test failures"

**Solution:**
```bash
# Run tests with verbose output
pytest -v

# Run specific failing test
pytest tests/test_name.py::test_function -v

# Check test output for details
pytest -v -s  # -s shows print statements

# Ensure .env is configured
cat .env  # Should have at least one API key
```

#### "ChromaDB persistence errors"

**Solution:**
```bash
# Reset vector database
rm -rf chroma_db/

# Re-run the application (will rebuild database)
python src/app.py
```

### Known Limitations

- **Document Size**: Maximum document size limited by available RAM
- **Chunk Size**: 512-character chunks may split complex concepts; adjust `chunking.chunk_size` in config.yaml
- **Embedding Quality**: Depends on sentence-transformers model choice
- **Answer Quality**: Depends on LLM provider and model selection
- **Rate Limits**: Free tiers have usage restrictions; query optimization doubles LLM calls per question
- **Context Window**: Top-3 chunks may miss relevant information; increase `rag.default_n_results` in config.yaml
- **Language Support**: Optimized for English; multilingual models available

---

## 📚 Additional Resources

### Official Documentation
- [ChromaDB Documentation](https://docs.trychroma.com/) - Vector database guide
- [LangChain Documentation](https://python.langchain.com/) - LLM framework
- [Sentence Transformers](https://www.sbert.net/) - Embedding models
- [OpenAI API Documentation](https://platform.openai.com/docs) - GPT models
- [Google Gemini API Documentation](https://ai.google.dev/docs) - Gemini models
- [Groq API Documentation](https://console.groq.com/docs) - Fast inference
- [DeepEval Documentation](https://docs.confident-ai.com/) - RAG evaluation

### Ready Tensor Resources
- [Agentic AI Essentials Certification](https://www.readytensor.ai/agentic-ai-essentials-cert/) - Course overview
- [AI Certification Programs](https://www.readytensor.ai/certifications/) - Additional certifications

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### What This Means

The MIT License allows you to:

- ✅ **Use** commercially
- ✅ **Modify** as needed
- ✅ **Distribute** freely
- ✅ **Sublicense** under different terms
- ✅ **Use privately** without restrictions

**Requirement:** Include the original copyright notice and license text.

**No Warranty:** Software provided "as is" without warranty.

---

## 🙏 Acknowledgments

This project uses the following excellent open-source libraries:

- **[LangChain](https://github.com/langchain-ai/langchain)** (MIT License) - LLM framework
- **[ChromaDB](https://github.com/chroma-core/chroma)** (Apache 2.0) - Vector database
- **[Sentence Transformers](https://github.com/UKPLab/sentence-transformers)** (Apache 2.0) - Embedding models
- **[DeepEval](https://github.com/confident-ai/deepeval)** (Apache 2.0) - RAG evaluation framework
- **[OpenAI API](https://platform.openai.com/)** - GPT models
- **[Groq API](https://groq.com/)** - Fast LLM inference
- **[Google Gemini API](https://ai.google.dev/)** - Gemini models
- **[Pytest](https://pytest.org/)** (MIT License) - Testing framework
- **[Python-dotenv](https://github.com/theskumar/python-dotenv)** (BSD-3-Clause) - Environment management
- **[PyYAML](https://github.com/yaml/pyyaml)** (MIT License) - YAML parsing
- **[NumPy](https://numpy.org/)** (BSD-3-Clause) - Numerical computing

Special thanks to the [Ready Tensor Agentic AI Essentials Certification Program](https://www.readytensor.ai/agentic-ai-essentials-cert/).

---

## 📞 Support & Contact

### Getting Help

- **Issues:** [Open an issue](https://github.com/david-001/agentic-ai-essentials-cert-project/issues) on GitHub
- **Documentation:** Check the README and code comments
- **Questions:** Reach out through GitHub discussions
- **Tests:** Run `pytest`

### Contributing

Contributions are welcome! Here's how to contribute:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes**
4. **Add tests** for new functionality
5. **Run the test suite** (`pytest`)
6. **Commit your changes** (`git commit -m 'Add amazing feature'`)
7. **Push to the branch** (`git push origin feature/amazing-feature`)
8. **Open a Pull Request**

**Contribution Guidelines:**
- Follow existing code style
- Add tests for new features
- Update documentation as needed
- Keep commits focused and descriptive
