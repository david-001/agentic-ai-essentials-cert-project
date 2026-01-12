# RAG-Based AI Assistant for Document Q&A

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://python.langchain.com/)

## 📑 Table of Contents

- [Project Summary](#-project-summary)
- [Project Overview](#-project-overview)
  - [What This Project Does](#what-this-project-does)
  - [How It Works](#how-it-works)
  - [Technical Approach](#technical-approach)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
  - [Prerequisites](#prerequisites)
  - [Step-by-Step Setup](#step-by-step-setup)
  - [Verify Installation](#verify-installation)
- [Usage](#-usage)
  - [Basic Usage](#basic-usage)
  - [Example Session](#example-session)
  - [Example Questions](#example-questions)
- [Configuration](#️-configuration)
- [Testing](#-testing)
  - [Test Categories](#test-categories)
  - [Running Tests](#running-tests)
  - [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)
- [Support & Contact](#-support--contact)

---

## 📋 Project Summary

An intelligent question-answering system that uses **Retrieval-Augmented Generation (RAG)** to provide accurate answers based on your own document collection. Built as part of the Agentic AI Essentials certification program, this assistant combines semantic search with large language models to create a personalized knowledge base that you can query in natural language.

**Key Capabilities:**
- 🔍 Semantic document search using vector embeddings
- 💬 Natural language question answering
- 📚 Support for multiple document formats (.txt, .md)
- 🤖 Multi-provider LLM support (OpenAI, Groq, Google Gemini)
- 💾 Persistent vector storage with ChromaDB
- ⚙️ YAML-based configuration for easy customization
- 🔄 Intelligent text chunking with RecursiveCharacterTextSplitter
- 🧪 Comprehensive test suite with performance metrics

---

## 🎯 Project Overview

### What This Project Does

This RAG assistant solves a common problem: **how to make AI understand and answer questions about YOUR specific documents**. Unlike general-purpose chatbots that only know information from their training data, this system:

1. **Ingests** your documents (company policies, research papers, documentation, etc.)
2. **Indexes** them using semantic embeddings for intelligent search
3. **Retrieves** the most relevant information when you ask a question
4. **Generates** accurate, context-aware answers using state-of-the-art LLMs

### How It Works

The system implements a complete RAG pipeline:

```
┌─────────────────┐
│  Your Documents │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│ Text Chunking   │────▶│ Create Embeddings│
│ (Intelligent)   │     │ (Vector Space)   │
└─────────────────┘     └────────┬─────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   Store in      │
                        │   ChromaDB      │
                        └────────┬────────┘
                                 │
         ┌───────────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  User Question  │────▶│ Semantic Search  │────▶│   LLM Generates  │
│                 │     │ (Find Top 3)     │     │   Answer with    │
└─────────────────┘     └──────────────────┘     │   Context        │
                                                 └──────────────────┘
```

**Step-by-Step Process:**

1. **Document Loading**: Reads documents from the `data/` directory
2. **Text Chunking**: Uses LangChain's RecursiveCharacterTextSplitter for intelligent chunking
3. **Embedding Creation**: Converts text chunks to 384-dimensional vectors using sentence-transformers
4. **Vector Storage**: Stores embeddings in ChromaDB for fast similarity search
5. **Query Processing**: When you ask a question:
   - Your question is converted to a vector
   - System finds the most similar document chunks
   - Chunks are combined as context for the LLM
   - LLM generates an answer based on the retrieved context

### Technical Approach

**Vector Search vs. Keyword Search:**
Traditional keyword search looks for exact word matches. This RAG system uses *semantic search*, which understands meaning. For example:
- Question: "How many days off do I get?"
- Matches documents about: "vacation policy", "paid time off", "leave entitlement"
- Even if the exact words "days off" don't appear

**Why RAG Instead of Fine-Tuning:**
- ✅ No expensive model retraining needed
- ✅ Update knowledge base by just adding documents
- ✅ Reduces hallucinations (answers based on your docs)
- ✅ Transparent and explainable results

---

## 📁 Repository Structure

```
agentic-ai-essentials-cert-project/
│
├── src/                          # Source code directory
│   ├── app.py                    # Main application with RAG pipeline
│   ├── vectordb.py               # Vector database wrapper for ChromaDB
│   └── config.py                 # Configuration loader (loads from YAML)
│
├── config/                       # Configuration directory
│   └── config.yaml              # YAML configuration file (edit settings here)
│
├── data/                         # Document collection
│   ├── api_documentation.md     # Sample: API documentation
│   ├── company_policies.md      # Sample: HR policies
│   ├── customer_faq.md          # Sample: FAQ
│   ├── product_documentation.md # Sample: Product info
│   └── security_compliance.md   # Sample: Security docs
│
├── tests/                        # Comprehensive test suite
│   ├── test_app.py              # Integration tests
│   ├── test_vectordb.py         # Vector database tests
│   ├── test_retrieval_quality.py # Retrieval performance tests
│   ├── test_generation_quality.py # Answer quality tests
│   ├── test_performance_metrics.py # Performance benchmarks
│   └── performance_reporter.py   # Metrics reporting utility
│
├── requirements.txt              # Python dependencies
├── pytest.ini                   # Test configuration
├── .env                         # Environment variables (API keys) - DO NOT COMMIT
├── .env.example                 # Example environment file (safe to share)
├── .gitignore                   # Git ignore rules
├── LICENSE                      # MIT License
└── README.md                    # This file
│
└── chroma_db/                   # Vector database storage (auto-created)
```

### Key Files Explained

| File/Directory | Purpose | When to Modify |
|----------------|---------|----------------|
| `src/app.py` | Main application entry point | Add features, modify prompt template |
| `src/vectordb.py` | Handles chunking, embedding, search | Adjust embedding logic |
| `src/config.py` | Loads configuration from YAML | Rarely (handles loading automatically) |
| `config/config.yaml` | **Main configuration file** | **Change settings here** |
| `.env` | API keys and secrets | Set your API keys here |
| `.env.example` | Template for environment variables | Reference for setup |
| `data/` | Your document collection | Add your .txt or .md files |
| `tests/` | Comprehensive test suite | Extend with new tests |
| `tests/performance_reporter.py` | **Performance metrics calculator** | Use for generating evaluation reports |
| `pytest.ini` | Test runner configuration | Modify test settings |

---

## 🌟 Features

- 📚 **Document Loading**: Automatically loads .txt and .md files from `data/` folder
- 🔍 **Semantic Search**: Uses sentence transformers for accurate document retrieval
- 💾 **Persistent Storage**: ChromaDB vector database with local persistence
- 🤖 **Multi-LLM Support**: Works with OpenAI, Groq, or Google Gemini
- 🔄 **Smart Chunking**: Uses RecursiveCharacterTextSplitter for context preservation
- ⚙️ **YAML Configuration**: Easy-to-edit configuration file
- 🎯 **Reproducible Results**: Fixed random seed for consistent behavior
- 🧪 **Comprehensive Testing**: Unit, integration, and quality tests
- 📊 **Performance Monitoring**: Built-in metrics and reporting with performance_reporter.py
- 📈 **Detailed Analytics**: Precision, Recall, MRR, NDCG tracking
- 📄 **Multiple Performance Report Formats**: JSON, CSV, and Markdown exports
- 🔐 **Environment-based Secrets**: Secure API key management

---

## 🚀 Installation

### Prerequisites

- **Python 3.10 or higher** (Python 3.10 or 3.11 recommended)
- **pip** (Python package installer)
- **One of these API keys** (at least one required):
  - OpenAI API key
  - Google Gemini API key
  - Groq API key

### Step-by-Step Setup

#### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd agentic-ai-essentials-cert-project
```

#### Step 2: Create Virtual Environment (Recommended)

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `langchain` (v0.3.27) - LLM orchestration framework
- `langchain-core` (v0.3.76) - Core LangChain functionality
- `langchain-openai`, `langchain-groq`, `langchain-google-genai` - LLM provider integrations
- `langchain-text-splitters` (v0.3.11) - Intelligent text chunking
- `chromadb` (v1.0.12) - Vector database
- `sentence-transformers` (v5.1.0) - Embedding models
- `python-dotenv` (v1.1.1) - Environment variable management
- `pyyaml` (v6.0.2) - YAML configuration support
- `pytest` (v9.0.2) - Testing framework
- `pytest-cov` (v7.0.0) - Test coverage reporting

**Installation time:** 2-3 minutes depending on internet speed

#### Step 4: Set Up Environment Variables

**IMPORTANT SECURITY NOTE:** Never commit your `.env` file to version control. It should already be in `.gitignore`.

Create a `.env` file with your API key:

```bash
# Copy the example
cp .env.example .env

# Edit with your favorite editor
nano .env
```

Add **at least one** API key to `.env`:

```bash
# OpenAI
OPENAI_API_KEY=sk-proj-your-key-here
OPENAI_MODEL=gpt-4o-mini

# OR Groq
GROQ_API_KEY=gsk-your-key-here
GROQ_MODEL=llama-3.1-8b-instant

# OR Google Gemini
GOOGLE_API_KEY=AIza-your-key-here
GOOGLE_MODEL=gemini-2.0-flash

# Embedding Configuration (usually no need to change)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector Database Configuration
CHROMA_COLLECTION_NAME=rag_documents
CHROMA_DB_PATH=./chroma_db

# LLM Configuration
DEFAULT_LLM_TEMPERATURE=0.0

# File Paths
DATA_DIRECTORY=data
```

**Where to get API keys:**
- **OpenAI:** https://platform.openai.com/api-keys
- **Groq:** https://console.groq.com/keys
- **Google Gemini:** https://makersuite.google.com/app/apikey

#### Step 5: Add Your Documents

Place your documents in the `data/` folder:

```bash
# The project includes 5 sample documents (.md format):
data/
  ├── api_documentation.md
  ├── company_policies.md
  ├── customer_faq.md
  ├── product_documentation.md
  └── security_compliance.md

# Add your own documents (supported formats: .txt, .md)
cp your_document.md data/
cp another_doc.txt data/
```

**Supported formats:** `.txt`, `.md`

**Document tips:**
- Keep documents focused on specific topics
- Use clear section headers for better chunking
- Aim for 1-50 pages per document
- Plain text works best

### Verify Installation

Run this to verify everything is set up correctly:

```bash
python src/app.py
```

If you see the interactive prompt, you're ready to go!

---

## 💻 Usage

### Basic Usage

1. **Run the assistant:**
   ```bash
   python src/app.py
   ```

2. **Ask questions about your documents:**
   ```
   Your question: What is the remote work policy?
   ```

3. **Exit when done:**
   ```
   Your question: quit
   ```

### Example Session

```bash
$ python src/app.py

Initializing RAG Assistant...
Using Google Gemini model: gemini-2.0-flash
Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
Vector database initialized with collection: rag_documents
RAG Assistant initialized successfully

Loading documents...
Loaded: api_documentation.md
Loaded: customer_faq.md
Loaded: company_policies.md
Loaded: security_compliance.md
Loaded: product_documentation.md
Loaded 5 sample documents
Processing 5 documents...
Document 1: Split into 33 chunks
Document 2: Split into 55 chunks
Document 3: Split into 37 chunks
Document 4: Split into 45 chunks
Document 5: Split into 40 chunks
Creating embeddings for 210 chunks...
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:02<00:00,  3.08it/s]
Adding to vector database...
Successfully added 210 chunks to vector database
Enter a question or 'quit' to exit: What vacation days do employees get?
Employees get vacation days based on their years of service:
*   0-2 years: 20 days (4 weeks) per year
*   3-5 years: 25 days (5 weeks) per year
*   6+ years: 30 days (6 weeks) per year
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
- "What are the rate limits?"

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

### YAML Configuration (`config/config.yaml`)

```yaml
# Embedding Model Configuration
embedding:
  model: sentence-transformers/all-MiniLM-L6-v2

# Vector Database Configuration
database:
  collection_name: rag_documents
  path: ./chroma_db

# LLM Configuration
llm:
  temperature: 0.0

# File Paths
paths:
  data_directory: data
```

### After Changing Configuration

If you change embedding model or chunking settings, delete the vector database to re-index:

```bash
rm -rf chroma_db/
python src/app.py
```

---

## 🧪 Testing

This project includes a comprehensive test suite with multiple categories of tests to ensure quality and performance.

### Test Categories

The test suite is organized into several categories using pytest markers:

1. **Unit Tests** (`@pytest.mark.unit`)
   - Individual component testing
   - Vector database operations
   - Document loading functions

2. **Integration Tests** (`@pytest.mark.integration`)
   - End-to-end RAG pipeline
   - Multi-component interactions
   - Complete workflow testing

3. **Retrieval Quality Tests** (`@pytest.mark.quality`)
   - Precision@k metrics
   - Recall@k metrics
   - Mean Reciprocal Rank (MRR)
   - NDCG@k scores

4. **Generation Quality Tests** (`@pytest.mark.generation`)
   - Answer relevance
   - Context usage
   - Factual accuracy
   - Response completeness

5. **Performance Tests** (`@pytest.mark.performance`)
   - Retrieval latency
   - Generation latency
   - End-to-end response time
   - Throughput metrics

### Running Tests

**Run all tests:**
```bash
pytest
```

**Run with verbose output:**
```bash
pytest -v
```

**Run specific test categories:**
```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Quality tests only
pytest -m quality

# Performance tests only
pytest -m performance

# Generation tests only
pytest -m generation
```

**Run specific test files:**
```bash
# Vector database tests
pytest tests/test_vectordb.py

# Application integration tests
pytest tests/test_app.py

# Retrieval quality tests
pytest tests/test_retrieval_quality.py

# Generation quality tests
pytest tests/test_generation_quality.py

# Performance metrics tests
pytest tests/test_performance_metrics.py
```

**Run with coverage report:**
```bash
pytest --cov=src --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`.

### Performance Metrics

The test suite includes a sophisticated performance reporting system that tracks:

**Retrieval Metrics:**
- **Precision@3**: Accuracy of top 3 retrieved documents
- **Recall@3**: Coverage of relevant documents in top 3
- **MRR (Mean Reciprocal Rank)**: Ranking quality of first relevant result
- **NDCG@5**: Normalized discounted cumulative gain at 5
- **Average Latency**: Retrieval speed in milliseconds

**Generation Metrics:**
- **Answer Relevance**: How well answers address the question
- **Context Precision**: Accuracy of retrieved context
- **Faithfulness**: Degree to which answers are grounded in context
- **Hallucination Rate**: Detection of fabricated information

### Performance Reporter Utility

The `tests/performance_reporter.py` module provides comprehensive performance evaluation capabilities:

**Calculated Metrics:**

*Retrieval Performance:*
- `precision_at_3`: Proportion of top-3 results that are relevant
- `recall_at_3`: Proportion of relevant docs retrieved in top-3
- `mrr`: Mean reciprocal rank (1/rank of first relevant result)
- `ndcg_at_5`: Normalized discounted cumulative gain at position 5
- `avg_latency_ms`: Average retrieval time in milliseconds

*Generation Quality:*
- `faithfulness`: % of answer tokens found in retrieved context
- `answer_relevance`: Semantic similarity between answer and question
- `hallucination_rate`: % of claims not supported by context
- `context_adherence`: How closely answer follows retrieved information

**Performance Reporter Usage:**
```bash
python tests/performance_reporter.py
```

**Performance Report Formats:**

- JSON format for programmatic access
- CSV format for spreadsheet analysis
- Markdown format for documentation


### Test Configuration

Tests are configured via `pytest.ini`:

```ini
[pytest]
# Test discovery
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Test paths
testpaths = tests

# Output options
addopts = -v --strict-markers --tb=short --disable-warnings

# Test markers
markers =
    unit: Unit tests
    integration: Integration tests
    generation: Generation tests
    performance: Performance tests
    quality: Quality tests
```

### Writing Your Own Tests

To add custom tests:

1. Create a new file in `tests/` following the `test_*.py` pattern
2. Use pytest fixtures from existing tests
3. Add appropriate markers: `@pytest.mark.unit`, `@pytest.mark.integration`, etc.
4. Run your new tests: `pytest tests/your_test_file.py -v`

Example test structure:
```python
import pytest
from src.app import RAGAssistant

@pytest.mark.unit
def test_document_loading():
    """Test that documents load correctly."""
    # Your test code here
    pass

@pytest.mark.integration
def test_full_pipeline():
    """Test the complete RAG pipeline."""
    # Your test code here
    pass
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### "Environment variable not set"
**Solution:**
- Create `.env` file in project root
- Add at least one API key:
  ```bash
  OPENAI_API_KEY=sk-your-key-here
  ```
- Remove placeholder text like `your_key_here`
- Ensure no spaces around the `=` sign
- **Never commit `.env` to version control**

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
```

#### "No documents found in data directory"
**Solution:**
- Add `.txt` or `.md` files to `data/` folder
- Check that files have content (not empty)
- Verify file permissions (should be readable)

#### "Rate limit exceeded"
**Solution:**
- Switch to Google Gemini (more generous free tier)
- Wait a few minutes and try again
- Upgrade your API plan

#### "Python version not supported"
**Solution:**
- Check your Python version: `python --version`
- Ensure you are using Python 3.10 or higher
- Python 3.10 or 3.11 recommended
- Create fresh virtual environment with correct Python version:
  ```bash
  python3.10 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

#### "ImportError: cannot import name 'config'"
**Solution:**
```bash
# Ensure you're in the correct directory
cd agentic-ai-essentials-cert-project

# Run from src directory
python src/app.py
```

#### "Test failures"
**Solution:**
```bash
# Run tests with verbose output to see details
pytest -v

# Run specific failing test
pytest tests/test_name.py::test_function

# Check test dependencies are installed
pip install pytest pytest-cov
```

### Getting Help

- Check the configuration: `python src/config.py`
- Verify API key (Check in .env)
- Check Python version: `python --version`
- Reinstall dependencies: `pip install --force-reinstall -r requirements.txt`

---


### Limitations

- Maximum document size limited by available RAM
- Search quality depends on embedding model
- Answer quality depends on LLM model choice
- Rate limits apply to free tiers

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
- **[Sentence Transformers](https://github.com/UKPLab/sentence-transformers)** (Apache 2.0) - Embeddings
- **[OpenAI API](https://platform.openai.com/)** - GPT models
- **[Groq API](https://groq.com/)** - Fast LLM inference
- **[Google Gemini API](https://ai.google.dev/)** - Gemini models
- **[Pytest](https://pytest.org/)** - Testing framework

Special thanks to the **Agentic AI Essentials Certification Program** for the learning framework and project structure.

---

## 📞 Support & Contact

### Getting Help

- **Issues:** [Open an issue](https://github.com/your-username/rag-assistant/issues) on GitHub
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

---


## 📚 Additional Resources

### Official Documentation
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [Groq API Documentation](https://console.groq.com/docs)

### Learning Resources
- [Python Best Practices](https://docs.python-guide.org/)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Vector Database Guide](https://www.pinecone.io/learn/vector-database/)
- [Pytest Documentation](https://docs.pytest.org/)

---

## 🎯 Quick Reference

### Common Commands

```bash
# ===== Setup =====
python3 -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows
pip install -r requirements.txt

# ===== Configuration =====
cp .env.example .env              # Create environment file
nano .env                         # Edit with your API key

# ===== Running =====
python src/app.py                     # Start interactive assistant

# ===== Testing =====
pytest                            # Run all tests
pytest -v                         # Verbose output
pytest -m unit                    # Unit tests only
pytest -m integration             # Integration tests only
pytest -m quality                 # Quality tests only
pytest -m performance             # Performance tests only
pytest -m generation              # Generation tests only
pytest --cov=src --cov-report=html  # Coverage report

# ==== Performance Report =======
python tests/performance_reporter.py  # Generate performance report

# ===== Document Management =====
cp my_doc.md data/               # Add document
ls -lh data/                     # List documents

# ===== Database Management =====
rm -rf chroma_db/                # Reset database (will rebuild)
du -sh chroma_db/                # Check database size

```


---

*Last Updated: January 2026*
