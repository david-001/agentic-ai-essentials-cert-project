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
  - [Value Proposition](#value-proposition)
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
│   ├── api_documentation.txt    # Sample: API documentation
│   ├── company_policies.txt     # Sample: HR policies
│   ├── customer_faq.txt         # Sample: FAQ
│   ├── product_documentation.txt # Sample: Product info
│   └── security_compliance.txt   # Sample: Security docs
│
├── requirements.txt              # Python dependencies
├── .env                         # Environment variables (API keys)
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
| `data/` | Your document collection | Add your .txt or .md files |

---

## 🌟 Features

- 📚 **Document Loading**: Automatically loads .txt and .md files from `data/` folder
- 🔍 **Semantic Search**: Uses sentence transformers for accurate document retrieval
- 💾 **Persistent Storage**: ChromaDB vector database with local persistence
- 🤖 **Multi-LLM Support**: Works with OpenAI, Groq, or Google Gemini
- 🔄 **Smart Chunking**: Uses RecursiveCharacterTextSplitter for context preservation
- ⚙️ **YAML Configuration**: Easy-to-edit configuration file
- 🎯 **Reproducible Results**: Fixed random seed for consistent behavior

---

## 🚀 Installation

### Prerequisites

- **Python 3.10 or higher**
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
- `langchain-core` - LLM orchestration framework
- `langchain-openai`, `langchain-groq`, `langchain-google-genai` - LLM provider integrations
- `langchain-text-splitters` - Intelligent text chunking
- `chromadb` - Vector database
- `sentence-transformers` - Embedding models
- `python-dotenv` - Environment variable management
- `pyyaml` - YAML configuration support

**Installation time:** 2-3 minutes depending on internet speed

#### Step 4: Set Up Environment Variables

Create a `.env` file with your API key:

```bash
# Copy the example (if you have one)
cp .env.example .env

# Or create manually
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
```

**Where to get API keys:**
- **OpenAI:** https://platform.openai.com/api-keys
- **Groq:** https://console.groq.com/keys
- **Google Gemini:** https://makersuite.google.com/app/apikey

#### Step 5: Add Your Documents

Place your documents in the `data/` folder:

```bash
# The project includes 5 sample documents:
# - api_documentation.md
# - company_policies.md
# - customer_faq.md
# - product_documentation.md
# - security_compliance.md

# To add your own:
cp your_document.txt data/
cp your_other_doc.md data/
```

**Supported formats:** `.txt` and `.md` files

### Verify Installation

Run a quick test to ensure everything is set up correctly:

```bash
cd src
python app.py
```

**Expected output:**
```
Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
Using OpenAI model: gpt-4o-mini
Loading documents from: data
Loaded 5 documents
Processing documents...
Vector database initialized

Welcome to the RAG Assistant!
Ask questions about your documents (type 'quit' to exit)

Your question: 
```

If you see this, installation was successful! ✅

---

## 💻 Usage

### Basic Usage

1. **Navigate to the src directory:**
   ```bash
   cd src
   ```

2. **Run the assistant:**
   ```bash
   python app.py
   ```

3. **Ask questions about your documents:**
   ```
   Your question: What is the remote work policy?
   ```

4. **Exit when done:**
   ```
   Your question: quit
   ```

### Example Session

```bash
$ cd src
$ python app.py

Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
Using OpenAI model: gpt-4o-mini
Vector database initialized

Welcome to the RAG Assistant!
Ask questions about your documents (type 'quit' to exit)

Your question: What vacation days do employees get?

Answer: Full-time employees receive 15-20 days of paid vacation annually,
depending on tenure. New employees start with 15 days, increasing to 20
days after 3 years of service. Vacation requests should be submitted at
least 2 weeks in advance through the HR portal.

Your question: What's the API rate limit?

Answer: The API rate limits vary by plan: Starter plan has 100 requests
per hour, Professional plan has 1,000 requests per hour, and Business
plan has 10,000 requests per hour. Enterprise customers can request
custom rate limits.

Your question: quit

Thank you for using the RAG Assistant!
```

### Example Questions

Try these sample questions with the included documents:

**Company Policies:**
- "What is the remote work policy?"
- "How many vacation days do employees get?"
- "What are the parental leave benefits?"
- "Does the company provide health insurance?"

**API Documentation:**
- "What's the API rate limit for the Professional plan?"
- "How do I authenticate with the API?"
- "What are the available endpoints?"
- "How do webhooks work?"

**Product Information:**
- "What pricing plans are available?"
- "What integrations does the product support?"
- "What are the system requirements?"
- "Is there a free trial?"

**Security & Compliance:**
- "Is the platform GDPR compliant?"
- "What encryption is used?"
- "What certifications does the company have?"
- "What is the data retention policy?"

---

## ⚙️ Configuration

Configuration is managed through `config/config.yaml`. This makes it easy to adjust settings without modifying code.

### Viewing Current Settings

```bash
# From the src directory:
cd src
python config.py
```

This displays all current configuration values.

### Main Configuration File: `config/config.yaml`

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

### Common Adjustments

#### Change Embedding Model

```yaml
embedding:
  # Faster, lighter (default)
  model: sentence-transformers/all-MiniLM-L6-v2
  
  # OR more accurate, heavier
  # model: sentence-transformers/all-mpnet-base-v2
```

#### Change Data Directory

```yaml
paths:
  data_directory: my_documents  # Change from 'data'
```

#### Adjust LLM Temperature

```yaml
llm:
  temperature: 0.0    # Deterministic (default)
  # temperature: 0.7  # More creative
```

### After Changing Configuration

If you change embedding model or chunking settings, delete the vector database to re-index:

```bash
rm -rf chroma_db/
cd src
python app.py
```

---

## 🔧 Troubleshooting

### Common Issues

#### "No valid API key found"
**Solution:**
- Create `.env` file in project root
- Add at least one API key:
  ```bash
  OPENAI_API_KEY=sk-your-key-here
  ```
- Remove placeholder text like `your_key_here`

#### "ModuleNotFoundError: No module named 'X'"
**Solution:**
```bash
pip install -r requirements.txt
```

#### "No documents found in data directory"
**Solution:**
- Add `.txt` or `.md` files to `data/` folder
- Check that files have content
- Verify file permissions

#### "Rate limit exceeded"
**Solution:**
- Switch to Google Gemini (more generous free tier)
- Wait a few minutes and try again
- Upgrade your API plan

#### "Python version not supported"
**Solution:**
- Check your Python version: `python --version`
- Ensure you are using Python 3.10 or higher (Python 3.10 or 3.11 recommended)
- Create fresh virtual environment with correct Python version


#### "ImportError: cannot import name 'config'"
**Solution:**
```bash
# Make sure you're running from the src directory
cd src
python app.py
```

### Getting Help

- Check the configuration: `python src/config.py`
- Verify API key: `echo $OPENAI_API_KEY` (Linux/Mac)
- Check Python version: `python --version`
- Reinstall dependencies: `pip install --force-reinstall -r requirements.txt`

---

## 💡 Tips for Best Results

1. **Document Quality**: Well-formatted, clear documents work best
2. **File Organization**: Keep related documents in the `data/` folder
3. **Question Phrasing**: Ask specific questions for better answers
4. **API Selection**: 
   - OpenAI
   - Google Gemini
   - Groq for speed
5. **Configuration**: Adjust `config/config.yaml` to tune performance

---

## 📊 Project Stats

- **Lines of Code:** ~500 (excluding documentation)
- **Number of Core Dependencies:** 10+ packages
- **Supported File Formats:** 2 (txt, md)
- **Supported LLM Providers:** 3 (OpenAI, Groq, Google)
- **Embedding Dimensions:** 384 (all-MiniLM-L6-v2)
- **Sample Documents:** 5 included (~15KB total)

---

## 🔒 Data Privacy & Security

### Privacy Features

- ✅ **Local Processing**: All documents processed locally
- ✅ **Local Storage**: Vector embeddings stored in `chroma_db/`
- ✅ **Minimal Data Sent**: Only query text and relevant chunks sent to LLM APIs
- ✅ **Full Documents Never Sent**: Your complete documents stay on your machine

### API Costs (Approximate)

| Provider | Model | Cost per Query | 100 Queries |
|----------|-------|----------------|-------------|
| OpenAI | gpt-4o-mini | ~$0.001 | ~$0.10 |
| Google | Gemini | Free tier | Free |
| Groq | Llama 3.1 | Free tier | Free |

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

Special thanks to the **Agentic AI Essentials Certification Program** for the learning framework.

---

## 📞 Support & Contact

### Getting Help

- **Issues:** [Open an issue](https://github.com/your-username/rag-assistant/issues) on GitHub
- **Documentation:** Check the README and code comments
- **Questions:** Reach out through GitHub discussions

### Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest enhancements  
- Submit pull requests
- Share improvements

---

## 🎓 Educational Context

This project is part of the **Agentic AI Essentials Certification Program - Module 1**.

### Learning Objectives Covered

1. ✅ **Document Loading** - Reading and parsing file formats
2. ✅ **Text Chunking** - Intelligent splitting with RecursiveCharacterTextSplitter
3. ✅ **Vector Embeddings** - Converting text to numerical representations
4. ✅ **Vector Databases** - ChromaDB storage and querying
5. ✅ **Semantic Search** - Finding relevant information by meaning
6. ✅ **RAG Architecture** - Combining retrieval with generation
7. ✅ **LLM Integration** - Working with multiple AI providers
8. ✅ **Production Patterns** - Configuration, error handling, modularity

### Skills Demonstrated

- Python programming with modern libraries
- AI/ML engineering fundamentals
- Vector database operations
- API integration and management
- YAML-based configuration
- Software architecture and design patterns
- Documentation and project organization

---

## 🔮 Future Enhancements

### Near-term
- [ ] Add support for PDF documents
- [ ] Implement conversation history/memory
- [ ] Add source citations in responses
- [ ] Create web UI with Streamlit
- [ ] Add document metadata filtering

### Mid-term
- [ ] Support for Word documents (.docx)
- [ ] Hybrid search (semantic + keyword)
- [ ] Re-ranking of search results
- [ ] Multiple language support
- [ ] Export/import knowledge bases

### Long-term
- [ ] Multi-user support with access control
- [ ] Real-time document updates
- [ ] Advanced analytics
- [ ] Custom embedding model fine-tuning
- [ ] Platform integrations (Slack, Teams)

---

## 📚 Additional Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [Groq API Documentation](https://console.groq.com/docs)
- [Python Best Practices](https://docs.python-guide.org/)

---

## 🎯 Quick Reference

### Common Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure
nano .env  # Add API key

# Run
cd src
python app.py

# Add documents
cp my_doc.txt data/

# Reset database
rm -rf chroma_db/

# View configuration
python src/config.py

# Run tests
pytest
```

### Directory Navigation

```bash
# Project structure
cd agentic-ai-essentials-cert-project  # Project root
cd src                                  # Source code
cd config                              # Configuration
cd data                                # Documents
```

---

*Last Updated: January 2026*
