"""
Unit tests for VectorDB component.
Tests document chunking, embedding generation, and vector search functionality.
"""

import pytest
import os
import sys
from typing import List

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vectordb import VectorDB


@pytest.fixture
def vector_db():
    """Fixture to create a VectorDB instance for testing."""
    return VectorDB(collection_name="test_collection")


@pytest.fixture
def sample_documents():
    """Fixture providing sample documents for testing."""
    return [
        {
            'content': 'This is a test document about machine learning. '
                      'Machine learning is a subset of artificial intelligence. '
                      'It enables computers to learn from data without explicit programming.',
            'metadata': {'source': 'test_doc_1.txt', 'category': 'ML'}
        },
        {
            'content': 'Python is a popular programming language. '
                      'It is widely used for data science and web development. '
                      'Python has a simple and readable syntax.',
            'metadata': {'source': 'test_doc_2.txt', 'category': 'Programming'}
        }
    ]


class TestTextChunking:
    """Test suite for text chunking functionality."""
    
    @pytest.mark.unit
    def test_chunk_text_basic(self, vector_db):
        """Test basic text chunking."""
        text = "This is a simple test. " * 50  # Create text longer than chunk size
        chunks = vector_db.chunk_text(text, chunk_size=100, chunk_overlap=20)
        
        assert len(chunks) > 0, "Should create at least one chunk"
        assert all(isinstance(chunk, str) for chunk in chunks), "All chunks should be strings"
    
    @pytest.mark.unit
    def test_chunk_text_preserves_content(self, vector_db):
        """Test that chunking preserves all content."""
        text = "Word1 Word2 Word3 Word4 Word5"
        chunks = vector_db.chunk_text(text, chunk_size=20, chunk_overlap=5)
        
        # Join chunks and verify all original words are present
        combined = " ".join(chunks)
        for word in text.split():
            assert word in combined, f"Word '{word}' should be preserved in chunks"
    
    @pytest.mark.unit
    def test_chunk_text_respects_size(self, vector_db):
        """Test that chunks respect the specified size."""
        text = "A " * 1000  # Long text
        chunk_size = 100
        chunks = vector_db.chunk_text(text, chunk_size=chunk_size, chunk_overlap=10)
        
        # Most chunks should be close to the specified size (allowing some variance)
        for chunk in chunks[:-1]:  # Exclude last chunk which may be shorter
            assert len(chunk) <= chunk_size * 1.5, f"Chunk too large: {len(chunk)} chars"
    
    @pytest.mark.unit
    def test_chunk_text_empty_string(self, vector_db):
        """Test chunking with empty string."""
        chunks = vector_db.chunk_text("", chunk_size=100)
        assert len(chunks) == 0 or chunks == [''], "Empty string should produce no chunks or empty chunk"
    
    @pytest.mark.unit
    def test_chunk_text_short_text(self, vector_db):
        """Test chunking with text shorter than chunk size."""
        text = "Short text"
        chunks = vector_db.chunk_text(text, chunk_size=100)
        
        assert len(chunks) == 1, "Short text should produce single chunk"
        assert chunks[0] == text, "Short text should remain unchanged"


class TestEmbedding:
    """Test suite for embedding generation."""
    
    @pytest.mark.unit
    def test_embedding_model_loaded(self, vector_db):
        """Test that embedding model is properly loaded."""
        assert vector_db.embedding_model is not None, "Embedding model should be loaded"
        assert hasattr(vector_db.embedding_model, 'encode'), "Model should have encode method"
    
    @pytest.mark.unit
    def test_embedding_dimensions(self, vector_db):
        """Test that embeddings have correct dimensions."""
        text = "Test sentence for embedding"
        embedding = vector_db.embedding_model.encode([text])
        
        assert embedding.shape[0] == 1, "Should have one embedding"
        assert embedding.shape[1] == 384, "Should have 384 dimensions (all-MiniLM-L6-v2)"
    
    @pytest.mark.unit
    def test_embedding_consistency(self, vector_db):
        """Test that same text produces same embedding."""
        text = "Consistency test"
        embedding1 = vector_db.embedding_model.encode([text])
        embedding2 = vector_db.embedding_model.encode([text])
        
        import numpy as np
        assert np.allclose(embedding1, embedding2), "Same text should produce same embedding"
    
    @pytest.mark.unit
    def test_embedding_different_texts(self, vector_db):
        """Test that different texts produce different embeddings."""
        text1 = "Machine learning is amazing"
        text2 = "Python programming is fun"
        
        embedding1 = vector_db.embedding_model.encode([text1])
        embedding2 = vector_db.embedding_model.encode([text2])
        
        import numpy as np
        assert not np.allclose(embedding1, embedding2), "Different texts should produce different embeddings"


class TestDocumentAddition:
    """Test suite for adding documents to vector database."""
    
    @pytest.mark.unit
    def test_add_documents_basic(self, vector_db, sample_documents):
        """Test basic document addition."""
        vector_db.add_documents(sample_documents)
        
        # Verify documents were added
        count = vector_db.collection.count()
        assert count > 0, "Documents should be added to collection"
    
    @pytest.mark.unit
    def test_add_documents_with_metadata(self, vector_db, sample_documents):
        """Test that metadata is preserved when adding documents."""
        vector_db.add_documents(sample_documents)
        
        # Search to retrieve documents with metadata
        results = vector_db.search("machine learning", n_results=1)
        
        assert len(results['metadatas']) > 0, "Should retrieve metadata"
        metadata = results['metadatas'][0]
        assert 'source' in metadata, "Metadata should contain source"
    
    @pytest.mark.unit
    def test_add_empty_documents(self, vector_db):
        """Test handling of empty document list."""
        count_before = vector_db.collection.count()
        vector_db.add_documents([])
        count_after = vector_db.collection.count()
        assert count_after == count_before, "Empty document list should not add anything"
    
    @pytest.mark.unit
    def test_add_documents_creates_chunks(self, vector_db):
        """Test that documents are split into chunks."""
        docs = [{'content': 'A ' * 1000, 'metadata': {'source': 'test.txt'}}]
        vector_db.add_documents(docs)
        
        count = vector_db.collection.count()
        assert count > 1, "Long document should be split into multiple chunks"


class TestVectorSearch:
    """Test suite for vector search functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_documents(self, vector_db, sample_documents):
        """Setup: Add sample documents before each test."""
        vector_db.add_documents(sample_documents)
        return vector_db
    
    @pytest.mark.unit
    def test_search_returns_results(self, vector_db):
        """Test that search returns results."""
        results = vector_db.search("machine learning", n_results=2)
        
        assert 'documents' in results, "Results should contain documents"
        assert 'metadatas' in results, "Results should contain metadatas"
        assert 'distances' in results, "Results should contain distances"
        assert 'ids' in results, "Results should contain ids"
    
    @pytest.mark.unit
    def test_search_relevance(self, vector_db):
        """Test that search returns relevant results."""
        results = vector_db.search("machine learning artificial intelligence", n_results=1)
        
        # First document is about ML/AI, so it should be most relevant
        assert len(results['documents']) > 0, "Should return at least one result"
        doc_text = results['documents'][0].lower()
        assert 'machine learning' in doc_text or 'artificial' in doc_text, \
            "Result should be relevant to query"
    
    @pytest.mark.unit
    def test_search_n_results(self, vector_db):
        """Test that search respects n_results parameter."""
        results = vector_db.search("test query", n_results=1)
        assert len(results['documents']) <= 1, "Should return at most n_results"
        
        results = vector_db.search("test query", n_results=5)
        assert len(results['documents']) <= 5, "Should return at most n_results"
    
    @pytest.mark.unit
    def test_search_with_no_matches(self, vector_db):
        """Test search behavior when no close matches exist."""
        # This will still return results (vector search always returns something)
        # but distances should be higher
        results = vector_db.search("quantum mechanics physics", n_results=1)
        
        assert len(results['documents']) > 0, "Should still return results"
        assert len(results['distances']) > 0, "Should have distance scores"
    
    @pytest.mark.unit
    def test_search_empty_query(self, vector_db):
        """Test search with empty query string."""
        results = vector_db.search("", n_results=1)
        
        # Should still work but may return arbitrary results
        assert 'documents' in results, "Should return result structure even for empty query"


class TestCollectionManagement:
    """Test suite for collection management."""
    
    @pytest.mark.unit
    def test_collection_created(self, vector_db):
        """Test that collection is created."""
        assert vector_db.collection is not None, "Collection should be created"
        assert vector_db.collection.name == "test_collection", "Collection should have correct name"
    
    @pytest.mark.unit
    def test_collection_persistence(self):
        """Test that collection persists across instances."""
        # Create first instance and add document
        db1 = VectorDB(collection_name="persist_test")
        docs = [{'content': 'Persistence test', 'metadata': {'source': 'test.txt'}}]
        db1.add_documents(docs)
        count1 = db1.collection.count()
        
        # Create second instance with same collection name
        db2 = VectorDB(collection_name="persist_test")
        count2 = db2.collection.count()
        
        assert count2 == count1, "Collection should persist across instances"


@pytest.mark.unit
def test_vectordb_initialization():
    """Test VectorDB initialization with default parameters."""
    db = VectorDB()
    
    assert db is not None, "VectorDB should initialize"
    assert db.collection_name is not None, "Should have collection name"
    assert db.embedding_model is not None, "Should have embedding model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
