# RAG-Based AI Assistant for Document Q&A

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
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
  - [Step-by-Step Setup](#step-1-clone-the-repository)
  - [Verify Installation](#verify-installation)
- [Usage](#-usage)
  - [Basic Usage](#basic-usage)
  - [Example Session](#example-session)
  - [Example Questions](#example-questions)
  - [Advanced Usage](#advanced-usage)
- [Configuration](#️-configuration)
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
│ (500 chars)     │     │ (Vector Space)   │
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
2. **Text Chunking**: Splits documents into ~500 character chunks with 50 character overlap
3. **Embedding Creation**: Converts text chunks to 384-dimensional vectors using sentence-transformers
4. **Vector Storage**: Stores embeddings in ChromaDB for fast similarity search
5. **Query Processing**: When you ask a question:
   - Your question is converted to a vector
   - System finds the 3 most similar document chunks
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
- ✅ Cites sources (you can see which chunks were used)
- ✅ Reduces hallucinations (answers based on your docs)
- ✅ Works with any LLM provider

### Value Proposition

**For Developers:**
- Learn how to build production RAG systems
- Understand vector databases and embeddings
- Gain experience with LangChain framework
- Portfolio piece demonstrating AI engineering skills

**For Organizations:**
- Build internal knowledge bases
- Automate customer support with company-specific info
- Create searchable documentation systems
- Reduce time spent searching for information

**For Students:**
- Complete certification project requirements
- Hands-on experience with cutting-edge AI
- Reusable template for future projects
- Understanding of modern AI architectures

---

## 📁 Repository Structure

```
rag-assistant/
├── app.py                      # Main application with RAG pipeline
├── app_with_retry.py          # Enhanced version with rate limit handling
├── vectordb.py                # Vector database wrapper for ChromaDB
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
├── README.md                  # This file
├── .env.example              # Environment variables template
│
├── data/                      # Place your documents here
│   ├── company_policies.txt  # Sample document
│   └── product_documentation.txt  # Sample document
│
├── chroma_db/                # Vector database storage (auto-created)
│
└── docs/                     # Additional documentation
    ├── SETUP_GUIDE.md       # Step-by-step setup instructions
    ├── IMPLEMENTATION_SUMMARY.md  # Detailed implementation guide
    ├── RATE_LIMIT_SOLUTIONS.md   # Troubleshooting guide
    ├── QUICK_FIX.md         # Common issues and fixes
    └── LICENSE_GUIDE.md     # License selection guide
```

### Key Files Explained

| File | Purpose | When to Modify |
|------|---------|----------------|
| `app.py` | Main application entry point | Add features, modify prompt template |
| `vectordb.py` | Handles chunking, embedding, search | Adjust chunk size, embedding model |
| `requirements.txt` | Python dependencies | Add new libraries |
| `.env` | API keys and configuration | Set your API keys |
| `data/` | Your document collection | Add your .txt or .md files |

---

- 📚 **Document Loading**: Automatically loads .txt and .md files from `data/` folder
- 🔍 **Semantic Search**: Uses sentence transformers for accurate document retrieval
- 💾 **Persistent Storage**: ChromaDB vector database with local persistence
- 🤖 **Multi-LLM Support**: Works with OpenAI, Groq, or Google Gemini
- 🔄 **Smart Chunking**: Overlapping text chunks for better context preservation

---

## 🚀 Installation

### Prerequisites

- **Python 3.8 or higher**
- **pip** (Python package installer)
- **One of these API keys** (at least one required):
  - OpenAI API key (recommended for production)
  - Google Gemini API key (recommended for free tier)
  - Groq API key (fastest, but rate-limited)

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd rag-assistant
```

### Step 2: Create Virtual Environment (Recommended)

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

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `langchain-core` - LLM orchestration framework
- `langchain-openai`, `langchain-groq`, `langchain-google-genai` - LLM provider integrations
- `chromadb` - Vector database
- `sentence-transformers` - Embedding models
- `python-dotenv` - Environment variable management

**Installation time:** 2-3 minutes depending on internet speed

### Step 4: Set Up Environment Variables

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API key:

**Option A: Google Gemini (Recommended - Free)**
```env
GOOGLE_API_KEY=AIza-your-key-here
GOOGLE_MODEL=gemini-2.0-flash
```
Get key at: https://aistudio.google.com/app/apikey

**Option B: OpenAI (Best Quality - Paid)**
```env
OPENAI_API_KEY=sk-proj-your-key-here
OPENAI_MODEL=gpt-4o-mini
```
Get key at: https://platform.openai.com/api-keys

**Option C: Groq (Fastest - Free with Limits)**
```env
GROQ_API_KEY=gsk_your-key-here
GROQ_MODEL=llama-3.1-8b-instant
```
Get key at: https://console.groq.com/keys

### Step 5: Add Your Documents

Place your .txt or .md files in the `data/` folder:

```bash
mkdir -p data
# Copy your documents to the data/ folder
```

**Supported formats:**
- `.txt` - Plain text files
- `.md` - Markdown files

**Document guidelines:**
- Use clear, well-formatted text
- Each file should contain cohesive content
- Aim for 500-5000 words per document
- Use descriptive filenames

### Verify Installation

```bash
python app.py
```

If successful, you should see:
```
Initializing RAG Assistant...
Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
Using [Provider] model: [model-name]
Vector database initialized with collection: rag_documents
RAG Assistant initialized successfully
```

---

## 💻 Usage

### Basic Usage

1. **Start the assistant:**
```bash
python app.py
```

2. **Wait for initialization** (first run downloads embedding model ~90MB)

3. **Ask questions** about your documents:
```
Your question: What is the remote work policy?
```

4. **Type 'quit' to exit**

### Example Session

```bash
$ python app.py

Initializing RAG Assistant...
Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
Using OpenAI model: gpt-4o-mini
Vector database initialized with collection: rag_documents
RAG Assistant initialized successfully

Loading documents...
Loaded: company_policies.txt
Loaded: product_documentation.txt
Loaded 2 sample documents
Processing 2 documents...
Document 1: Split into 8 chunks
Document 2: Split into 10 chunks
Creating embeddings for 18 chunks...
Adding to vector database...
Successfully added 18 chunks to vector database

==================================================
RAG Assistant is ready!
==================================================
You can now ask questions about your documents.
Type 'quit' to exit.

Your question: What is the remote work policy?

Thinking...

Answer: According to the company policy, employees are eligible to work 
remotely up to 3 days per week with manager approval. Remote work arrangements 
must be documented and reviewed quarterly. Employees must maintain regular 
communication and be available during core business hours (10 AM - 3 PM local time).

--------------------------------------------------

Your question: quit
Goodbye!
```

### Example Questions

**For Company Policies:**
- "What is the vacation policy?"
- "How many vacation days do employees get?"
- "What are the core working hours?"
- "What is the professional development budget?"
- "How do I request remote work?"

**For Product Documentation:**
- "What are the pricing plans?"
- "What security features are available?"
- "How do I share files with external users?"
- "What are the system requirements?"
- "How much storage do I get?"

### Advanced Usage

**Adjust number of retrieved chunks:**
```python
# In app.py, modify the invoke() call:
result = assistant.invoke(question, n_results=5)  # Default is 3
```

**Change chunk size:**
```python
# In vectordb.py, modify chunk_text():
def chunk_text(self, text: str, chunk_size: int = 800):  # Default is 500
```

**Switch embedding model:**
```python
# In .env file:
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2  # More accurate but slower
```

**Use retry logic for rate limits:**
```bash
# Use the version with automatic retry
python app_with_retry.py
```

---

## ⚙️ Configuration

### Chunk Size

Modify in `vectordb.py`:

```python
def chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50)
```

- **chunk_size**: Characters per chunk (default: 500)
- **chunk_overlap**: Overlapping characters (default: 50)

### Number of Retrieved Chunks

Modify in `app.py` when calling `query()`:

```python
result = assistant.query(question, n_results=3)  # Change 3 to desired number
```

### Embedding Model

Change in `.env`:

```env
# Faster, lighter model (default)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# More accurate, heavier model
# EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

## Supported LLM Providers

### OpenAI (Recommended for best quality)
```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini  # or gpt-4o, gpt-4-turbo
```

### Groq (Fastest, free tier available)
```env
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.1-8b-instant  # or llama-3.1-70b-versatile
```

### Google Gemini (Good balance)
```env
GOOGLE_API_KEY=AIza...
GOOGLE_MODEL=gemini-2.0-flash  # or gemini-1.5-pro
```

## Project Structure

```
rag-assistant/
├── app.py                 # Main application and RAG pipeline
├── vectordb.py           # Vector database wrapper
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── .env                 # Your API keys (create this)
├── data/                # Your documents go here
│   ├── document1.txt
│   └── document2.md
└── chroma_db/           # Vector database storage (auto-created)
```

## Implementation Details

### Key Functions

#### `load_documents()` in `app.py`
- Reads all .txt and .md files from `data/` folder
- Returns list of documents with content and metadata

#### `chunk_text()` in `vectordb.py`
- Splits text into overlapping chunks
- Preserves context at chunk boundaries
- Uses word-based splitting for clean breaks

#### `add_documents()` in `vectordb.py`
- Chunks each document
- Creates embeddings using sentence transformers
- Stores in ChromaDB with metadata

#### `search()` in `vectordb.py`
- Creates query embedding
- Performs similarity search in ChromaDB
- Returns top N most relevant chunks

#### `query()` in `app.py`
- Retrieves relevant chunks via search
- Builds context from retrieved chunks
- Generates answer using LLM chain

## Troubleshooting

### "No valid API key found"
- Make sure you've created `.env` file (copy from `.env.example`)
- Add at least one API key
- Remove the `your_key_here` placeholder text

### "No documents found"
- Check that `data/` folder exists
- Add .txt or .md files to the folder
- Make sure files have content

### "ModuleNotFoundError"
- Run `pip install -r requirements.txt`
- Use a virtual environment (recommended)

### Poor answers
- Try increasing `n_results` for more context
- Adjust `chunk_size` for better/worse granularity
- Use a more powerful LLM model
- Add more relevant documents

## Advanced: Adding PDF Support

Uncomment in `requirements.txt`:
```
pypdf2==3.0.1
```

Add to `load_documents()` in `app.py`:
```python
elif filename.endswith('.pdf'):
    import PyPDF2
    with open(filepath, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        content = ""
        for page in pdf_reader.pages:
            content += page.extract_text()
        results.append({
            'content': content,
            'metadata': {'source': filename}
        })
```

## Tips for Best Results

1. **Document Quality**: Well-formatted, clear documents work best
2. **Chunk Size**: Smaller chunks (300-500) for specific facts, larger (800-1200) for concepts
3. **Overlap**: 10-20% overlap helps maintain context
4. **Number of Results**: Start with 3-5, adjust based on answer quality
5. **Prompt Engineering**: Modify the prompt template in `app.py` for your use case

## Next Steps

- Add support for more file types (.pdf, .docx, .csv)
- Implement metadata filtering for targeted search
- Add conversation history for multi-turn dialogues
- Create a web interface with Streamlit or Gradio
- Add citation tracking to show source documents

## Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs)
- [Groq API](https://console.groq.com/docs)

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for full details.

### What This Means

The MIT License is one of the most permissive open-source licenses. You are free to:

- ✅ **Use** this project for commercial purposes
- ✅ **Modify** the code to fit your needs
- ✅ **Distribute** your modified versions
- ✅ **Sublicense** under different terms
- ✅ **Use privately** without any restrictions

**The only requirement:** Include the original copyright notice and license text in any substantial portion of the software.

**No Warranty:** The software is provided "as is" without warranty of any kind. The authors are not liable for any damages or issues arising from its use.

### Why MIT License?

We chose the MIT License because:
- It's simple and easy to understand
- It's compatible with virtually all other licenses
- It allows maximum freedom for users
- It's the industry standard for educational projects
- It encourages collaboration and learning

---

## 🙏 Acknowledgments

This project was created as part of the **Agentic AI Essentials Certification Program** and uses the following excellent open-source libraries:

- **[LangChain](https://github.com/langchain-ai/langchain)** (MIT License) - LLM application framework
- **[ChromaDB](https://github.com/chroma-core/chroma)** (Apache 2.0) - Vector database
- **[Sentence Transformers](https://github.com/UKPLab/sentence-transformers)** (Apache 2.0) - Embedding models
- **[OpenAI API](https://platform.openai.com/)** - GPT models
- **[Groq API](https://groq.com/)** - Fast LLM inference
- **[Google Gemini API](https://ai.google.dev/)** - Gemini models

Special thanks to the Anthropic team and the Agentic AI Essentials program for providing the learning framework that made this project possible.

---

## 📞 Support & Contact

### Getting Help

- **Documentation:** Check the `docs/` folder for detailed guides
  - `SETUP_GUIDE.md` - Step-by-step installation
  - `IMPLEMENTATION_SUMMARY.md` - Code explanations
  - `RATE_LIMIT_SOLUTIONS.md` - Troubleshooting common issues
  
- **Issues:** Found a bug? [Open an issue](https://github.com/your-username/rag-assistant/issues)
- **Questions:** Have questions about the code? Check the implementation summary or reach out

### Contributing

While this is primarily an educational project, contributions are welcome! Feel free to:
- Report bugs
- Suggest enhancements
- Submit pull requests
- Share your improvements

---

## 🎓 Educational Context

This project is part of the **Agentic AI Essentials Certification Program - Module 1** and demonstrates:

### Learning Objectives Covered

1. ✅ **Document Loading** - Reading and parsing various file formats
2. ✅ **Text Chunking** - Splitting text while preserving context
3. ✅ **Vector Embeddings** - Converting text to numerical representations
4. ✅ **Vector Databases** - Storing and querying embeddings efficiently
5. ✅ **Semantic Search** - Finding relevant information by meaning, not keywords
6. ✅ **RAG Architecture** - Combining retrieval with generation
7. ✅ **LLM Integration** - Working with multiple AI providers
8. ✅ **Production Patterns** - Error handling, configuration, modularity

### Skills Demonstrated

- Python programming with modern libraries
- AI/ML engineering fundamentals
- Vector database operations
- API integration and management
- Software architecture and design patterns
- Documentation and project organization
- Open-source best practices

---

## 🔮 Future Enhancements

Potential improvements for this project:

### Near-term (Easy)
- [ ] Add support for PDF documents
- [ ] Implement conversation history/memory
- [ ] Add source citations in responses
- [ ] Create web UI with Streamlit
- [ ] Add document metadata filtering

### Mid-term (Moderate)
- [ ] Support for Word documents (.docx)
- [ ] Hybrid search (semantic + keyword)
- [ ] Re-ranking of search results
- [ ] Multiple language support
- [ ] Export/import knowledge bases

### Long-term (Advanced)
- [ ] Multi-user support with access control
- [ ] Real-time document updates
- [ ] Advanced analytics and usage tracking
- [ ] Custom fine-tuning of embedding models
- [ ] Integration with popular platforms (Slack, Teams)

---

## 📊 Project Stats

- **Lines of Code:** ~500 (excluding documentation)
- **Number of Dependencies:** 8 core packages
- **Supported File Formats:** 2 (txt, md)
- **Supported LLM Providers:** 3 (OpenAI, Groq, Google)
- **Default Chunk Size:** 500 characters
- **Default Overlap:** 50 characters
- **Embedding Dimensions:** 384 (all-MiniLM-L6-v2)

---

## ⚠️ Important Notes

### Data Privacy
- All documents are processed locally
- Vector embeddings are stored locally in `chroma_db/`
- Only query text and retrieved chunks are sent to LLM APIs
- Your full documents are never sent to external APIs

### API Costs
- **OpenAI gpt-4o-mini:** ~$0.001 per query (~$0.10 for 100 queries)
- **Google Gemini:** Free tier (generous limits)
- **Groq:** Free tier (limited requests per minute)

### Limitations
- Maximum document size: Limited by available RAM
- Chunk size affects answer quality vs. context length trade-off
- Search quality depends on embedding model
- Answer quality depends on LLM model choice

---

## 🎯 Quick Reference

### Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API key

# Run the assistant
python app.py

# Run with retry logic (for rate limits)
python app_with_retry.py

# Add new documents
cp your_document.txt data/

# Clear vector database (fresh start)
rm -rf chroma_db/
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Rate limit error | Switch to Google Gemini or use `app_with_retry.py` |
| No API key found | Check `.env` file exists and has valid key |
| Import errors | Run `pip install -r requirements.txt` |
| No documents found | Add .txt or .md files to `data/` folder |
| Slow first run | First run downloads embedding model (~90MB) |

---

## 📚 Additional Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [RAG Architecture Overview](https://www.anthropic.com/research/retrieval-augmented-generation)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [Groq API Documentation](https://console.groq.com/docs)

---

**Built with ❤️ for the Agentic AI Essentials Certification Program**

*Last Updated: January 2026*
