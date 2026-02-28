"""
Integration tests for RAG Assistant.
Tests the complete RAG pipeline including document loading, retrieval, and generation.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import RAGAssistant, load_documents


@pytest.fixture
def mock_llm():
    """Fixture providing a mock LLM for testing."""
    mock = MagicMock()
    mock.invoke.return_value = "This is a test answer based on the provided context."
    return mock


@pytest.fixture
def sample_documents():
    """Fixture providing sample documents for testing."""
    return [
        {
            'content': (
                "Company Vacation Policy\n\n"
                "Employees receive vacation days based on tenure:\n"
                "- Entry-level (0-2 years): 20 days per year\n"
                "- Mid-level (3-5 years): 25 days per year\n"
                "- Senior-level (5+ years): 30 days per year\n"
                "Vacation must be requested 2 weeks in advance."
            ),
            'metadata': {'source': 'company_policies.txt'}
        },
        {
            'content': (
                "API Authentication\n\n"
                "Our API supports two authentication methods:\n"
                "1. API Key Authentication - Include X-API-Key header\n"
                "2. OAuth 2.0 - Use for user-specific access\n"
                "Rate limits: 1000 requests/hour for API keys."
            ),
            'metadata': {'source': 'api_documentation.txt'}
        }
    ]


class TestRAGAssistantInitialization:
    """Test suite for RAG Assistant initialization."""
    
    @pytest.mark.integration
    def test_assistant_initialization_with_api_key(self):
        """Test assistant initializes with valid API key."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            assistant = RAGAssistant()
            assert assistant is not None, "Assistant should initialize"
            assert assistant.llm is not None, "LLM should be initialized"
            assert assistant.vector_db is not None, "Vector DB should be initialized"
    
    @pytest.mark.integration
    def test_assistant_initialization_without_api_key(self):
        """Test assistant fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No valid API key found"):
                RAGAssistant()
    
    @pytest.mark.integration
    def test_prompt_template_created(self):
        """Test that prompt template is created."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            assistant = RAGAssistant()
            assert assistant.prompt_template is not None, "Prompt template should be created"
    
    @pytest.mark.integration
    def test_chain_created(self):
        """Test that LangChain chain is created."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            assistant = RAGAssistant()
            assert assistant.chain is not None, "Chain should be created"


class TestDocumentLoading:
    """Test suite for document loading functionality."""
    
    @pytest.mark.integration
    def test_load_documents_from_directory(self, tmp_path):
        """Test loading documents from a directory."""
        # Create temporary test files
        (tmp_path / "test1.txt").write_text("Test document 1 content")
        (tmp_path / "test2.md").write_text("Test document 2 content")
        
        # Mock the config.DATA_DIRECTORY
        with patch('app.config.DATA_DIRECTORY', str(tmp_path)):
            docs = load_documents()
            
            assert len(docs) == 2, "Should load both documents"
            assert all('content' in doc for doc in docs), "All docs should have content"
            assert all('metadata' in doc for doc in docs), "All docs should have metadata"
    
    @pytest.mark.integration
    def test_load_documents_filters_extensions(self, tmp_path):
        """Test that only .txt and .md files are loaded."""
        (tmp_path / "valid.txt").write_text("Valid document")
        (tmp_path / "valid.md").write_text("Valid document")
        (tmp_path / "invalid.pdf").write_text("Should not load")
        (tmp_path / "also_invalid.docx").write_text("Should not load")
        
        with patch('app.config.DATA_DIRECTORY', str(tmp_path)):
            docs = load_documents()
            
            assert len(docs) == 2, "Should only load .txt or .md file"
    
    @pytest.mark.integration
    def test_load_documents_empty_directory(self, tmp_path):
        """Test loading from empty directory."""
        with patch('app.config.DATA_DIRECTORY', str(tmp_path)):
            docs = load_documents()
            
            assert len(docs) == 0, "Should return empty list for empty directory"
    
    @pytest.mark.integration
    def test_load_documents_preserves_metadata(self, tmp_path):
        """Test that document metadata is preserved."""
        (tmp_path / "source.txt").write_text("Content")
        
        with patch('app.config.DATA_DIRECTORY', str(tmp_path)):
            docs = load_documents()
            
            assert docs[0]['metadata']['source'] == 'source.txt', \
                "Metadata should contain source filename"


class TestRAGPipeline:
    """Test suite for complete RAG pipeline."""
    
    @pytest.fixture
    def assistant_with_docs(self, sample_documents):
        """Fixture providing RAG assistant with sample documents."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            assistant = RAGAssistant()
            assistant.add_documents(sample_documents)
            return assistant
    
    @pytest.mark.integration
    def test_add_documents(self, sample_documents):
        """Test adding documents to assistant."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            assistant = RAGAssistant()
            assistant.add_documents(sample_documents)
            
            # Verify documents were added
            count = assistant.vector_db.collection.count()
            assert count > 0, "Documents should be added to vector DB"
    
    @pytest.mark.integration
    def test_query_retrieves_context(self, assistant_with_docs):
        """Test that query retrieves relevant context."""
        # Mock both LLM chains to avoid actual API calls
        with patch.object(assistant_with_docs, 'query_optimization_chain') as mock_opt_chain, \
             patch.object(assistant_with_docs, 'chain') as mock_chain:
            mock_opt_chain.invoke.return_value = "vacation days employees tenure"
            mock_chain.invoke.return_value = "Mocked answer"
            
            # Perform query
            query = "How many vacation days do employees get?"
            assistant_with_docs.query(query)
            
            # Verify chain was invoked with context
            assert mock_chain.invoke.called, "Chain should be invoked"
            call_args = mock_chain.invoke.call_args[0][0]
            assert 'context' in call_args, "Should provide context to chain"
            assert 'question' in call_args, "Should provide question to chain"
    
    @pytest.mark.integration
    def test_query_returns_string(self, assistant_with_docs):
        """Test that query returns a string answer."""
        with patch.object(assistant_with_docs, 'query_optimization_chain') as mock_opt_chain, \
             patch.object(assistant_with_docs, 'chain') as mock_chain:
            mock_opt_chain.invoke.return_value = "test question optimized"
            mock_chain.invoke.return_value = "Test answer"
            
            result = assistant_with_docs.query("Test question")
            
            assert isinstance(result, str), "Query should return string"
            assert len(result) > 0, "Answer should not be empty"
    
    @pytest.mark.integration
    def test_query_context_relevance(self, assistant_with_docs):
        """Test that retrieved context is relevant to query."""
        # Query about vacation policy
        with patch.object(assistant_with_docs, 'query_optimization_chain') as mock_opt_chain, \
             patch.object(assistant_with_docs, 'chain') as mock_chain:
            mock_opt_chain.invoke.return_value = "vacation days policy employee tenure"
            mock_chain.invoke.return_value = "Mocked answer"
            
            assistant_with_docs.query("vacation policy")
            
            # Check the context passed to the chain
            call_args = mock_chain.invoke.call_args[0][0]
            context = call_args['context'].lower()
            
            # Context should mention vacation-related terms
            assert 'vacation' in context or 'days' in context, \
                "Context should be relevant to vacation query"
    
    @pytest.mark.integration
    def test_query_n_results_parameter(self, assistant_with_docs):
        """Test that n_results parameter controls retrieval."""
        with patch.object(assistant_with_docs, 'query_optimization_chain') as mock_opt_chain, \
             patch.object(assistant_with_docs.vector_db, 'search') as mock_search, \
             patch.object(assistant_with_docs, 'chain') as mock_chain:
            mock_opt_chain.invoke.return_value = "test optimized"
            mock_search.return_value = {
                'documents': ['doc1'], 
                'metadatas': [{}], 
                'distances': [0.1], 
                'ids': ['1']
            }
            mock_chain.invoke.return_value = "Answer"
            
            # Query with custom n_results
            assistant_with_docs.query("test", n_results=5)
            
            # Verify search was called with correct n_results
            assert mock_search.called, "Search should be called"
            call_kwargs = mock_search.call_args[1]
            assert call_kwargs.get('n_results') == 5, \
                "Should use custom n_results parameter"


class TestLLMProviderSelection:
    """Test suite for LLM provider selection."""
    
    @pytest.mark.integration
    def test_openai_provider_selected(self):
        """Test OpenAI provider is selected when key is present."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key', 'OPENAI_MODEL': 'gpt-4o-mini'}):
            assistant = RAGAssistant()
            # Verify it's using OpenAI (would need to check instance type in real scenario)
            assert assistant.llm is not None, "Should initialize with OpenAI"
    
    @pytest.mark.integration
    def test_groq_provider_fallback(self):
        """Test Groq provider is used when OpenAI key is absent."""
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-key'}, clear=True):
            assistant = RAGAssistant()
            assert assistant.llm is not None, "Should initialize with Groq"
    
    @pytest.mark.integration
    def test_google_provider_fallback(self):
        """Test Google provider is used when other keys are absent."""
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-key'}, clear=True):
            assistant = RAGAssistant()
            assert assistant.llm is not None, "Should initialize with Google"


class TestErrorHandling:
    """Test suite for error handling."""
    
    @pytest.mark.integration
    def test_query_with_empty_string(self):
        """Test handling of empty query string."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            assistant = RAGAssistant()
            
            with patch.object(assistant, 'query_optimization_chain') as mock_opt_chain, \
                 patch.object(assistant, 'chain') as mock_chain:
                mock_opt_chain.invoke.return_value = ""
                mock_chain.invoke.return_value = "Default response"
                
                result = assistant.query("")
                
                assert isinstance(result, str), "Should return string even for empty query"
    
    @pytest.mark.integration
    def test_query_with_no_documents(self):
        """Test querying when no documents are loaded."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            assistant = RAGAssistant()
            
            with patch.object(assistant, 'query_optimization_chain') as mock_opt_chain, \
                 patch.object(assistant, 'chain') as mock_chain:
                mock_opt_chain.invoke.return_value = "test query optimized"
                mock_chain.invoke.return_value = "No context answer"
                
                # Should still work but with no context
                result = assistant.query("test query")
                
                assert isinstance(result, str), "Should return answer even with no documents"


@pytest.mark.integration
def test_end_to_end_workflow(sample_documents, tmp_path):
    """Test complete end-to-end workflow."""
    # Setup: Create test documents
    for i, doc in enumerate(sample_documents):
        (tmp_path / f"doc{i}.txt").write_text(doc['content'])
    
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        with patch('app.config.DATA_DIRECTORY', str(tmp_path)):
            # Load documents
            docs = load_documents()
            assert len(docs) > 0, "Should load documents"
            
            # Initialize assistant
            assistant = RAGAssistant()
            
            # Add documents
            assistant.add_documents(docs)
            
            # Mock both LLM chains to avoid API calls
            with patch.object(assistant, 'query_optimization_chain') as mock_opt_chain, \
                 patch.object(assistant, 'chain') as mock_chain:
                mock_opt_chain.invoke.return_value = "test query optimized"
                mock_chain.invoke.return_value = "Test answer"
                
                # Query
                result = assistant.query("test query")
                
                assert isinstance(result, str), "Should return answer"
                assert len(result) > 0, "Answer should not be empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =============================================================================
# Tests added to address feedback: query optimization & RAG domain scope
# =============================================================================

class TestQueryOptimization:
    """Test suite for LLM-driven query optimization."""

    @pytest.fixture
    def assistant(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            return RAGAssistant()

    @pytest.mark.integration
    def test_optimize_query_called_before_search(self, assistant):
        """_optimize_query must run before vector_db.search."""
        with patch.object(assistant, '_optimize_query', wraps=assistant._optimize_query) as mock_opt, \
             patch.object(assistant.vector_db, 'search', return_value={
                 'documents': ['ctx'], 'metadatas': [{'document_type': 'FAQ', 'source': 'f.md'}],
                 'distances': [0.1], 'ids': ['1']
             }) as mock_search, \
             patch.object(assistant, 'chain') as mock_chain:
            mock_chain.invoke.return_value = "answer"
            mock_opt.return_value = "expanded query"

            assistant.query("how do I reset password")

            mock_opt.assert_called_once()
            # search must use the optimized string, not the raw input
            call_query = mock_search.call_args[0][0]
            assert call_query == "expanded query", \
                "vector_db.search should receive the optimized query, not the raw input"

    @pytest.mark.integration
    def test_optimize_query_returns_string(self, assistant):
        """_optimize_query must return a non-empty string."""
        with patch.object(assistant, 'query_optimization_chain') as mock_chain:
            mock_chain.invoke.return_value = "  two-factor authentication login TaskFlow Pro  "
            result = assistant._optimize_query("how do I use 2FA")
            assert isinstance(result, str), "_optimize_query should return a string"
            assert len(result.strip()) > 0, "_optimize_query result should not be empty"
            # Leading/trailing whitespace should be stripped
            assert result == result.strip(), "_optimize_query should strip whitespace"

    @pytest.mark.integration
    def test_optimize_query_receives_domain_context(self, assistant):
        """_optimize_query must pass domain and document_types to the LLM chain."""
        with patch.object(assistant, 'query_optimization_chain') as mock_chain:
            mock_chain.invoke.return_value = "optimized"
            assistant._optimize_query("test question")
            call_kwargs = mock_chain.invoke.call_args[0][0]
            assert 'domain' in call_kwargs, "domain must be included in query optimization prompt"
            assert 'document_types' in call_kwargs, "document_types must be included in query optimization prompt"
            assert 'question' in call_kwargs, "question must be passed to query optimization chain"

    @pytest.mark.integration
    def test_query_pipeline_uses_optimized_query_for_retrieval(self, assistant):
        """End-to-end: optimized query (not raw input) must reach the vector search."""
        raw = "whats the refund policy"
        optimized = "TaskFlow Pro refund cancellation billing policy"

        with patch.object(assistant, '_optimize_query', return_value=optimized) as mock_opt, \
             patch.object(assistant.vector_db, 'search', return_value={
                 'documents': [], 'metadatas': [], 'distances': [], 'ids': []
             }) as mock_search, \
             patch.object(assistant, 'chain') as mock_chain:
            mock_chain.invoke.return_value = "answer"
            assistant.query(raw)

            mock_opt.assert_called_once_with(raw)
            mock_search.assert_called_once()
            search_query_arg = mock_search.call_args[0][0]
            assert search_query_arg == optimized, \
                "Search must use the optimized query string"


class TestDocumentDomainIntegration:
    """Test suite for domain-specific document loading and metadata."""

    @pytest.mark.integration
    def test_load_documents_assigns_document_type(self, tmp_path):
        """Known filenames must receive domain-specific document_type metadata."""
        (tmp_path / "customer_faq.md").write_text("FAQ content")
        (tmp_path / "api_documentation.md").write_text("API content")

        with patch('app.config.DATA_DIRECTORY', str(tmp_path)):
            docs = load_documents()

        types = {d['metadata']['source']: d['metadata']['document_type'] for d in docs}
        assert types.get('customer_faq.md') == "Customer FAQ", \
            "customer_faq.md should be typed as 'Customer FAQ'"
        assert types.get('api_documentation.md') == "API Documentation", \
            "api_documentation.md should be typed as 'API Documentation'"

    @pytest.mark.integration
    def test_load_documents_assigns_domain(self, tmp_path):
        """Every loaded document must carry a 'domain' metadata field."""
        (tmp_path / "some_doc.md").write_text("content")

        with patch('app.config.DATA_DIRECTORY', str(tmp_path)):
            docs = load_documents()

        for doc in docs:
            assert 'domain' in doc['metadata'], "Each document must have a 'domain' metadata key"
            assert "TaskFlow" in doc['metadata']['domain'], \
                "domain should reference TaskFlow Pro"

    @pytest.mark.integration
    def test_load_documents_unknown_file_gets_fallback_type(self, tmp_path):
        """Files not in DOCUMENT_TYPE_MAP should get a 'General Document' type."""
        (tmp_path / "random_notes.md").write_text("some notes")

        with patch('app.config.DATA_DIRECTORY', str(tmp_path)):
            docs = load_documents()

        assert docs[0]['metadata']['document_type'] == "General Document", \
            "Unknown filenames should fall back to 'General Document'"

    @pytest.mark.integration
    def test_all_known_document_types_covered(self, tmp_path):
        """All five TaskFlow Pro document types should be resolvable."""
        from app import DOCUMENT_TYPE_MAP
        for stem, info in DOCUMENT_TYPE_MAP.items():
            (tmp_path / f"{stem}.md").write_text(f"Content for {stem}")

        with patch('app.config.DATA_DIRECTORY', str(tmp_path)):
            docs = load_documents()

        returned_types = {d['metadata']['document_type'] for d in docs}
        expected_types = {info['type'] for info in DOCUMENT_TYPE_MAP.values()}
        assert expected_types == returned_types, \
            "All defined document types should be assigned when their files are present"

    @pytest.mark.integration
    def test_format_context_includes_document_type(self):
        """_format_context must include the document type in each context section."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            assistant = RAGAssistant()

        search_results = {
            'documents': ['Auth info here', 'Vacation days info here'],
            'metadatas': [
                {'document_type': 'API Documentation', 'source': 'api_documentation.md'},
                {'document_type': 'Company Policies', 'source': 'company_policies.md'},
            ],
            'distances': [0.1, 0.2],
            'ids': ['1', '2'],
        }
        context = assistant._format_context(search_results)
        assert 'API Documentation' in context, "Context should reference document type"
        assert 'Company Policies' in context, "Context should reference second document type"

    @pytest.mark.integration
    def test_rag_prompt_receives_domain_and_document_types(self):
        """The answer chain must be invoked with domain and document_types fields."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            assistant = RAGAssistant()

        with patch.object(assistant, '_optimize_query', return_value="optimized"), \
             patch.object(assistant.vector_db, 'search', return_value={
                 'documents': ['ctx'], 'metadatas': [{'document_type': 'FAQ', 'source': 'f.md'}],
                 'distances': [0.1], 'ids': ['1']
             }), \
             patch.object(assistant, 'chain') as mock_chain:
            mock_chain.invoke.return_value = "answer"
            assistant.query("test")

            call_kwargs = mock_chain.invoke.call_args[0][0]
            assert 'domain' in call_kwargs, "chain.invoke must receive 'domain'"
            assert 'document_types' in call_kwargs, "chain.invoke must receive 'document_types'"
            assert 'context' in call_kwargs, "chain.invoke must receive 'context'"
            assert 'question' in call_kwargs, "chain.invoke must receive 'question'"
            assert "TaskFlow" in call_kwargs['domain'], "domain must reference TaskFlow Pro"
