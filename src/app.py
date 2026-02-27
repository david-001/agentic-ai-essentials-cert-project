import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import config

# Load environment variables
load_dotenv()

# ============================================================================
# Domain Configuration: TaskFlow Pro document types and their search keywords
# ============================================================================
DOCUMENT_DOMAIN = "TaskFlow Pro - Project Management Platform"

DOCUMENT_TYPE_MAP = {
    "customer_faq": {
        "type": "Customer FAQ",
        "description": "Frequently asked questions from TaskFlow Pro customers",
        "keywords": ["faq", "question", "how do i", "how to", "can i", "support"],
    },
    "product_documentation": {
        "type": "Product Documentation",
        "description": "Technical and feature documentation for TaskFlow Pro",
        "keywords": ["feature", "integration", "api", "setup", "configure", "install"],
    },
    "company_policies": {
        "type": "Company Policies",
        "description": "TaskFlow Pro employee handbook and internal policies",
        "keywords": ["policy", "vacation", "benefits", "employee", "handbook", "hr"],
    },
    "api_documentation": {
        "type": "API Documentation",
        "description": "Developer API reference and authentication guides",
        "keywords": ["api", "endpoint", "authentication", "oauth", "request", "response"],
    },
    "security_compliance": {
        "type": "Security & Compliance",
        "description": "Security practices, compliance certifications, and data policies",
        "keywords": ["security", "compliance", "gdpr", "soc2", "encryption", "data"],
    },
}


def _get_document_metadata(filename: str) -> Dict:
    """
    Resolve domain-specific metadata for a document based on its filename.

    Matches the filename stem against the DOCUMENT_TYPE_MAP to assign a
    structured type, description, and keyword hints. Falls back to a generic
    category when the filename does not match a known type.

    Args:
        filename: The basename of the document file (e.g. 'customer_faq.md')

    Returns:
        A metadata dict containing 'source', 'domain', 'document_type',
        'document_description', and 'search_keywords'.
    """
    stem = os.path.splitext(filename)[0].lower()
    doc_info = DOCUMENT_TYPE_MAP.get(stem, {
        "type": "General Document",
        "description": "TaskFlow Pro reference document",
        "keywords": [],
    })
    return {
        "source": filename,
        "domain": DOCUMENT_DOMAIN,
        "document_type": doc_info["type"],
        "document_description": doc_info["description"],
        "search_keywords": ", ".join(doc_info["keywords"]),
    }


def load_documents() -> List[str]:
    """
    Load documents from the configured data directory.

    Reads all .txt and .md files from config.DATA_DIRECTORY and attaches
    domain-specific metadata (document type, description, search keywords)
    to each document using the TaskFlow Pro DOCUMENT_TYPE_MAP.

    Returns:
        List of document dicts, each with 'content' and 'metadata' keys.
    """
    results = []
    data_dir = config.DATA_DIRECTORY

    if not os.path.exists(data_dir):
        print(f"Warning: {data_dir} directory not found. Creating it...")
        os.makedirs(data_dir)
        print(f"Please add your documents to the '{data_dir}' folder and run again.")
        return results

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)

        if filename.endswith('.txt') or filename.endswith('.md'):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Attach domain-specific metadata
                metadata = _get_document_metadata(filename)

                results.append({'content': content, 'metadata': metadata})
                print(f"Loaded [{metadata['document_type']}]: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    if len(results) == 0:
        print(f"\nNo documents found in '{data_dir}' folder.")
        print("Please add some .txt or .md files to get started.")

    return results


class RAGAssistant:
    """
    A RAG-based AI assistant for the TaskFlow Pro document domain.

    Implements query optimization via LLM-driven query expansion before
    vector retrieval, and uses a domain-aware prompt template that grounds
    all answers in the TaskFlow Pro knowledge base.

    Supports OpenAI, Groq, and Google Gemini LLM providers.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize vector database
        self.vector_db = VectorDB()

        # ----------------------------------------------------------------
        # Query Optimization Prompt
        # Rewrites and expands a user query into a retrieval-optimised form
        # tailored to the TaskFlow Pro document domain.
        # ----------------------------------------------------------------
        query_optimization_template = """You are a search query optimizer for a {domain} knowledge base.

The knowledge base contains these document types:
{document_types}

Your job is to rewrite the user's question into an optimized search query that will
retrieve the most relevant chunks from the knowledge base.

Rules:
- Expand abbreviations and acronyms (e.g. "2FA" -> "two-factor authentication")
- Add domain-specific synonyms that are likely to appear in the documents
- Remove filler words and focus on key concepts
- Keep the rewritten query concise (1-2 sentences maximum)
- Do NOT answer the question -- only rewrite it for retrieval

Original question: {question}

Optimized search query:"""

        self.query_optimization_prompt = ChatPromptTemplate.from_template(
            query_optimization_template
        )
        self.query_optimization_chain = (
            self.query_optimization_prompt | self.llm | StrOutputParser()
        )

        # ----------------------------------------------------------------
        # RAG Answer Prompt
        # Domain-aware template that instructs the LLM to answer strictly
        # from retrieved TaskFlow Pro context and cite document sources.
        # ----------------------------------------------------------------
        rag_template = """You are a knowledgeable assistant for {domain}.

You have access to the following document types in the knowledge base:
{document_types}

Use ONLY the retrieved context below to answer the question.
- If the context contains a direct answer, provide it clearly and concisely.
- Use bullet points for lists, steps, or multiple items.
- Always mention which document type the information comes from (e.g. "According to the API Documentation...").
- If the question cannot be answered from the provided context, respond with:
  "This information is not available in the TaskFlow Pro knowledge base. Please contact support."
- Never use outside knowledge beyond what is in the context.

Retrieved Context:
{context}

Question: {question}

Answer:"""

        self.prompt_template = ChatPromptTemplate.from_template(rag_template)
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=model_name,
                temperature=config.DEFAULT_LLM_TEMPERATURE,
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=model_name,
                temperature=config.DEFAULT_LLM_TEMPERATURE,
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=config.DEFAULT_LLM_TEMPERATURE,
            )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

    def _get_domain_context(self) -> Dict[str, str]:
        """
        Build the domain context strings used in both prompt templates.

        Returns:
            Dict with 'domain' and 'document_types' keys.
        """
        doc_types_list = "\n".join(
            f"  - {info['type']}: {info['description']}"
            for info in DOCUMENT_TYPE_MAP.values()
        )
        return {"domain": DOCUMENT_DOMAIN, "document_types": doc_types_list}

    def _optimize_query(self, user_query: str) -> str:
        """
        Rewrite and expand the user's raw query for better vector retrieval.

        Uses the LLM to produce a retrieval-optimised version of the query
        that includes domain synonyms, expands abbreviations, and focuses on
        the key concepts most likely to appear in TaskFlow Pro documents.

        Args:
            user_query: The original question entered by the user.

        Returns:
            An optimized query string for use with vector_db.search().
        """
        domain_ctx = self._get_domain_context()
        optimized = self.query_optimization_chain.invoke({
            "domain": domain_ctx["domain"],
            "document_types": domain_ctx["document_types"],
            "question": user_query,
        })
        optimized = optimized.strip()
        print(f"  [Query Optimization] Original: '{user_query}'")
        print(f"  [Query Optimization] Optimized: '{optimized}'")
        return optimized

    def _format_context(self, search_results: Dict) -> str:
        """
        Format retrieved chunks into a structured context string.

        Includes the document type and source from metadata alongside the
        chunk text so the LLM can attribute answers to the correct source.

        Args:
            search_results: The dict returned by VectorDB.search().

        Returns:
            A formatted multi-section context string.
        """
        if not search_results.get('documents'):
            return "No relevant information found in the knowledge base."

        sections = []
        for idx, (doc, meta) in enumerate(
            zip(search_results['documents'], search_results['metadatas']), start=1
        ):
            doc_type = meta.get('document_type', 'Document')
            source = meta.get('source', 'unknown')
            sections.append(
                f"[Source {idx} -- {doc_type} ({source})]:\n{doc}"
            )

        return "\n\n".join(sections)

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of document dicts with 'content' and 'metadata'.
        """
        self.vector_db.add_documents(documents)

    def query(self, input: str, n_results: int = 3) -> str:
        """
        Query the RAG assistant using the two-stage optimized pipeline.

        Stage 1 -- Query Optimization:
            The raw user input is rewritten by the LLM into a retrieval-
            optimised query with domain synonyms and expanded terms.

        Stage 2 -- Retrieval & Generation:
            The optimised query is used to search the vector database.
            Retrieved chunks are formatted with source attribution and passed
            to the domain-aware answer prompt to generate the final response.

        Args:
            input: User's question.
            n_results: Number of relevant chunks to retrieve.

        Returns:
            A string answer grounded in the TaskFlow Pro knowledge base.
        """
        # Stage 1: Optimize the query for retrieval
        optimized_query = self._optimize_query(input)

        # Stage 2: Retrieve relevant context using the optimized query
        search_results = self.vector_db.search(optimized_query, n_results=n_results)

        # Format context with source attribution
        context = self._format_context(search_results)

        # Generate the answer using the domain-aware prompt
        domain_ctx = self._get_domain_context()
        llm_answer = self.chain.invoke({
            "domain": domain_ctx["domain"],
            "document_types": domain_ctx["document_types"],
            "context": context,
            "question": input,
        })

        return llm_answer


def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        print("\nLoading documents...")
        sample_docs = load_documents()
        print(f"Loaded {len(sample_docs)} documents")

        assistant.add_documents(sample_docs)

        done = False
        while not done:
            question = input("\nEnter a question or 'quit' to exit: ")
            if question.lower() == "quit":
                done = True
            else:
                result = assistant.query(question)
                print(f"\n{result}")

    except Exception as e:
        print(f"Error running RAG assistant: {e}")


if __name__ == "__main__":
    main()
