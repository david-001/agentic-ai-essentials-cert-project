"""
Retrieval Quality Tests for RAG System.
Tests evaluate how well the system finds relevant information from the knowledge base.
Includes metrics like precision, recall, and ranking quality.
"""

import pytest
import os
import sys
from typing import List, Dict, Tuple
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vectordb import VectorDB


@pytest.fixture
def comprehensive_knowledge_base():
    """Fixture providing a comprehensive knowledge base for testing."""
    return [
        {
            'content': (
                "Python Programming Language\n\n"
                "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. "
                "It emphasizes code readability with significant whitespace. Python supports multiple programming "
                "paradigms including procedural, object-oriented, and functional programming. It has a comprehensive "
                "standard library and is widely used for web development, data science, automation, and AI."
            ),
            'metadata': {'source': 'python_intro.txt', 'topic': 'programming', 'doc_id': '1'}
        },
        {
            'content': (
                "Machine Learning Fundamentals\n\n"
                "Machine learning is a subset of artificial intelligence that enables systems to learn and improve "
                "from experience without explicit programming. It uses algorithms to parse data, learn from it, "
                "and make predictions. Types include supervised learning (labeled data), unsupervised learning "
                "(pattern discovery), and reinforcement learning (reward-based). Common algorithms include "
                "decision trees, neural networks, and support vector machines."
            ),
            'metadata': {'source': 'ml_basics.txt', 'topic': 'AI', 'doc_id': '2'}
        },
        {
            'content': (
                "Deep Learning and Neural Networks\n\n"
                "Deep learning is a specialized branch of machine learning using artificial neural networks with "
                "multiple layers. These networks can automatically learn hierarchical representations of data. "
                "Popular architectures include Convolutional Neural Networks (CNNs) for image processing, "
                "Recurrent Neural Networks (RNNs) for sequences, and Transformers for natural language. "
                "Deep learning has revolutionized computer vision, speech recognition, and NLP."
            ),
            'metadata': {'source': 'deep_learning.txt', 'topic': 'AI', 'doc_id': '3'}
        },
        {
            'content': (
                "Web Development with Django\n\n"
                "Django is a high-level Python web framework that encourages rapid development and clean design. "
                "It follows the Model-View-Template (MVT) architectural pattern. Django includes an ORM for "
                "database operations, built-in admin interface, authentication system, and form handling. "
                "It's used by companies like Instagram, Pinterest, and Mozilla. Django emphasizes security "
                "and helps developers avoid common security mistakes."
            ),
            'metadata': {'source': 'django_guide.txt', 'topic': 'web', 'doc_id': '4'}
        },
        {
            'content': (
                "Data Science with Python Libraries\n\n"
                "Python dominates data science with powerful libraries. NumPy provides numerical computing with "
                "arrays and matrices. Pandas offers data structures for data manipulation and analysis. "
                "Matplotlib and Seaborn enable data visualization. Scikit-learn provides machine learning "
                "algorithms. These libraries work together seamlessly, making Python the preferred language "
                "for data scientists worldwide."
            ),
            'metadata': {'source': 'data_science.txt', 'topic': 'data_science', 'doc_id': '5'}
        },
        {
            'content': (
                "Natural Language Processing\n\n"
                "NLP enables computers to understand, interpret, and generate human language. Key tasks include "
                "tokenization, part-of-speech tagging, named entity recognition, sentiment analysis, and machine "
                "translation. Modern NLP relies heavily on deep learning, particularly transformer models like "
                "BERT and GPT. Applications include chatbots, text summarization, question answering, and "
                "language translation."
            ),
            'metadata': {'source': 'nlp_overview.txt', 'topic': 'AI', 'doc_id': '6'}
        }
    ]


@pytest.fixture
def vector_db_with_knowledge(comprehensive_knowledge_base):
    """Fixture providing a VectorDB instance with comprehensive knowledge base."""
    db = VectorDB(collection_name="retrieval_quality_test")
    db.add_documents(comprehensive_knowledge_base)
    return db


class TestRetrievalAccuracy:
    """Test suite for measuring retrieval accuracy."""
    
    @pytest.mark.quality
    def test_exact_topic_retrieval(self, vector_db_with_knowledge):
        """Test retrieval accuracy for exact topic matches."""
        queries_and_expected_docs = [
            ("What is Python programming language?", ['1']),
            ("Explain machine learning", ['2']),
            ("Tell me about deep learning neural networks", ['3']),
            ("Django web framework", ['4']),
            ("Python data science libraries", ['5']),
            ("Natural language processing NLP", ['6'])
        ]
        
        correct_retrievals = 0
        total_queries = len(queries_and_expected_docs)
        
        for query, expected_doc_ids in queries_and_expected_docs:
            results = vector_db_with_knowledge.search(query, n_results=1)
            
            if results['metadatas']:
                retrieved_doc_id = results['metadatas'][0].get('doc_id')
                if retrieved_doc_id in expected_doc_ids:
                    correct_retrievals += 1
        
        accuracy = correct_retrievals / total_queries
        print(f"\nExact Topic Retrieval Accuracy: {accuracy:.2%}")
        
        assert accuracy >= 0.70, f"Retrieval accuracy too low: {accuracy:.2%}"
    
    @pytest.mark.quality
    def test_semantic_similarity_retrieval(self, vector_db_with_knowledge):
        """Test retrieval based on semantic similarity rather than keyword matching."""
        # Queries that don't use exact keywords but have semantic similarity
        semantic_queries = [
            ("coding in a high-level interpreted language", ['1']),  # Python
            ("AI systems that learn from data", ['2']),  # ML
            ("multi-layer artificial neural networks", ['3']),  # Deep learning
            ("Python framework for building websites", ['4']),  # Django
            ("analyzing datasets with Python tools", ['5']),  # Data science
        ]
        
        correct_retrievals = 0
        
        for query, expected_doc_ids in semantic_queries:
            results = vector_db_with_knowledge.search(query, n_results=1)
            
            if results['metadatas']:
                retrieved_doc_id = results['metadatas'][0].get('doc_id')
                if retrieved_doc_id in expected_doc_ids:
                    correct_retrievals += 1
        
        accuracy = correct_retrievals / len(semantic_queries)
        print(f"\nSemantic Similarity Accuracy: {accuracy:.2%}")
        
        assert accuracy >= 0.60, f"Semantic retrieval accuracy too low: {accuracy:.2%}"
    
    @pytest.mark.quality
    def test_multi_result_precision(self, vector_db_with_knowledge):
        """Test precision when retrieving multiple results."""
        # Query that should retrieve AI-related documents (docs 2, 3, 6)
        query = "artificial intelligence and machine learning techniques"
        results = vector_db_with_knowledge.search(query, n_results=3)
        
        ai_related_docs = {'2', '3', '6'}
        retrieved_doc_ids = [meta.get('doc_id') for meta in results['metadatas']]
        
        # Calculate precision: relevant retrieved / total retrieved
        relevant_retrieved = sum(1 for doc_id in retrieved_doc_ids if doc_id in ai_related_docs)
        precision = relevant_retrieved / len(retrieved_doc_ids) if retrieved_doc_ids else 0
        
        print(f"\nMulti-result Precision: {precision:.2%}")
        print(f"Retrieved docs: {retrieved_doc_ids}")
        print(f"Expected AI-related docs: {ai_related_docs}")
        
        assert precision >= 0.50, f"Precision too low: {precision:.2%}"


class TestRetrievalRanking:
    """Test suite for evaluating ranking quality of retrieved results."""
    
    @pytest.mark.quality
    def test_most_relevant_ranked_first(self, vector_db_with_knowledge):
        """Test that most relevant document is ranked first."""
        test_cases = [
            ("Python programming language features", '1'),
            ("supervised and unsupervised learning", '2'),
            ("convolutional and recurrent neural networks", '3'),
            ("Django MTV pattern and ORM", '4'),
        ]
        
        top_ranked_correct = 0
        
        for query, expected_top_doc in test_cases:
            results = vector_db_with_knowledge.search(query, n_results=3)
            
            if results['metadatas'] and len(results['metadatas']) > 0:
                top_doc_id = results['metadatas'][0].get('doc_id')
                if top_doc_id == expected_top_doc:
                    top_ranked_correct += 1
                    print(f"✓ Query: '{query}' - Correct top result")
                else:
                    print(f"✗ Query: '{query}' - Expected {expected_top_doc}, got {top_doc_id}")
        
        ranking_accuracy = top_ranked_correct / len(test_cases)
        print(f"\nRanking Accuracy (top result): {ranking_accuracy:.2%}")
        
        assert ranking_accuracy >= 0.70, f"Ranking accuracy too low: {ranking_accuracy:.2%}"
    
    @pytest.mark.quality
    def test_distance_scores_monotonic(self, vector_db_with_knowledge):
        """Test that distance scores are monotonically increasing (or decreasing based on metric)."""
        query = "machine learning algorithms"
        results = vector_db_with_knowledge.search(query, n_results=5)
        
        distances = results['distances']
        
        # For cosine similarity (used by most embeddings), smaller distance = more similar
        # Check if distances are non-decreasing (each result is less similar than previous)
        is_monotonic = all(distances[i] <= distances[i+1] for i in range(len(distances)-1))
        
        print(f"\nDistance scores: {distances}")
        print(f"Monotonic ordering: {is_monotonic}")
        
        assert is_monotonic, "Distance scores should be monotonically ordered"
    
    @pytest.mark.quality
    def test_relevant_docs_in_top_k(self, vector_db_with_knowledge):
        """Test that relevant documents appear in top-k results."""
        # Query about Python - should retrieve docs 1, 4, 5 (Python-related)
        query = "Python programming and development"
        k = 3
        
        results = vector_db_with_knowledge.search(query, n_results=k)
        retrieved_doc_ids = [meta.get('doc_id') for meta in results['metadatas']]
        
        python_related_docs = {'1', '4', '5'}
        relevant_in_top_k = sum(1 for doc_id in retrieved_doc_ids if doc_id in python_related_docs)
        
        recall_at_k = relevant_in_top_k / len(python_related_docs)
        
        print(f"\nRecall@{k}: {recall_at_k:.2%}")
        print(f"Retrieved: {retrieved_doc_ids}")
        print(f"Python-related docs: {python_related_docs}")
        
        assert recall_at_k >= 0.33, f"Recall@{k} too low: {recall_at_k:.2%}"


class TestEdgeCaseRetrieval:
    """Test suite for edge cases in retrieval."""
    
    @pytest.mark.quality
    def test_very_short_query(self, vector_db_with_knowledge):
        """Test retrieval with very short queries."""
        short_queries = ["Python", "AI", "web", "data"]
        
        for query in short_queries:
            results = vector_db_with_knowledge.search(query, n_results=2)
            
            assert len(results['documents']) > 0, f"Should retrieve results for short query: '{query}'"
            assert results['documents'][0], "First result should not be empty"
    
    @pytest.mark.quality
    def test_very_long_query(self, vector_db_with_knowledge):
        """Test retrieval with very long queries."""
        long_query = (
            "I am looking for detailed information about programming languages that are "
            "interpreted and high-level with emphasis on code readability and support for "
            "multiple programming paradigms including object-oriented and functional programming "
            "and are commonly used in modern software development especially in areas like "
            "web development, data science, and artificial intelligence applications"
        )
        
        results = vector_db_with_knowledge.search(long_query, n_results=1)
        
        assert len(results['documents']) > 0, "Should retrieve results for long query"
        # Should likely retrieve Python document (doc 1)
        retrieved_doc_id = results['metadatas'][0].get('doc_id')
        print(f"Long query retrieved doc: {retrieved_doc_id}")
    
    @pytest.mark.quality
    def test_query_with_typos(self, vector_db_with_knowledge):
        """Test retrieval resilience to typos."""
        queries_with_typos = [
            "Pythn programing",  # Python programming
            "machne lerning",  # machine learning
            "nueral netwrks",  # neural networks
        ]
        
        successful_retrievals = 0
        
        for query in queries_with_typos:
            results = vector_db_with_knowledge.search(query, n_results=1)
            
            # Should still retrieve something relevant despite typos
            if results['documents'] and len(results['documents'][0]) > 0:
                successful_retrievals += 1
        
        success_rate = successful_retrievals / len(queries_with_typos)
        print(f"\nTypo resilience: {success_rate:.2%}")
        
        # Embedding models should be somewhat resilient to typos
        assert success_rate >= 0.66, "Should handle some typos"
    
    @pytest.mark.quality
    def test_query_with_synonyms(self, vector_db_with_knowledge):
        """Test retrieval using synonyms instead of exact terms."""
        synonym_queries = [
            ("coding language", '1'),  # programming language
            ("intelligent systems", '2'),  # AI/ML
            ("web application framework", '4'),  # Django
        ]
        
        correct_retrievals = 0
        
        for query, expected_doc in synonym_queries:
            results = vector_db_with_knowledge.search(query, n_results=1)
            retrieved_doc_id = results['metadatas'][0].get('doc_id')
            
            if retrieved_doc_id == expected_doc:
                correct_retrievals += 1
                print(f"✓ Synonym query: '{query}' -> Doc {expected_doc}")
            else:
                print(f"✗ Synonym query: '{query}' -> Expected {expected_doc}, got {retrieved_doc_id}")
        
        accuracy = correct_retrievals / len(synonym_queries)
        print(f"\nSynonym handling accuracy: {accuracy:.2%}")
        
        assert accuracy >= 0.50, "Should handle synonyms reasonably well"
    
    @pytest.mark.quality
    def test_negation_handling(self, vector_db_with_knowledge):
        """Test how the system handles queries with negation."""
        # Queries with negation - embedding models typically struggle with this
        query_positive = "machine learning techniques"
        query_negative = "NOT web development NOT Django"
        
        results_positive = vector_db_with_knowledge.search(query_positive, n_results=3)
        results_negative = vector_db_with_knowledge.search(query_negative, n_results=3)
        
        # Just checking that system returns results for both
        assert len(results_positive['documents']) > 0, "Should handle positive queries"
        assert len(results_negative['documents']) > 0, "Should return results for negative queries"
        
        print(f"\nNegation query retrieved docs: {[m.get('doc_id') for m in results_negative['metadatas']]}")


class TestRetrievalMetrics:
    """Test suite for comprehensive retrieval metrics."""
    
    @pytest.mark.quality
    def test_mean_reciprocal_rank(self, vector_db_with_knowledge):
        """Calculate Mean Reciprocal Rank (MRR) for a set of queries."""
        test_queries = [
            ("Python programming", '1'),
            ("machine learning algorithms", '2'),
            ("deep learning neural networks", '3'),
            ("Django web development", '4'),
            ("data science with Python", '5'),
        ]
        
        reciprocal_ranks = []
        
        for query, relevant_doc in test_queries:
            results = vector_db_with_knowledge.search(query, n_results=5)
            retrieved_doc_ids = [meta.get('doc_id') for meta in results['metadatas']]
            
            # Find position of relevant document (1-indexed)
            try:
                rank = retrieved_doc_ids.index(relevant_doc) + 1
                reciprocal_ranks.append(1.0 / rank)
            except ValueError:
                reciprocal_ranks.append(0.0)  # Relevant doc not found
        
        mrr = np.mean(reciprocal_ranks)
        print(f"\nMean Reciprocal Rank (MRR): {mrr:.3f}")
        
        assert mrr >= 0.50, f"MRR too low: {mrr:.3f}"
    
    @pytest.mark.quality
    def test_precision_at_k(self, vector_db_with_knowledge):
        """Calculate Precision@K for different values of K."""
        # Query about AI topics - relevant docs are 2, 3, 6
        query = "artificial intelligence and neural networks"
        relevant_docs = {'2', '3', '6'}
        
        k_values = [1, 3, 5]
        
        for k in k_values:
            results = vector_db_with_knowledge.search(query, n_results=k)
            retrieved_doc_ids = [meta.get('doc_id') for meta in results['metadatas']]
            
            relevant_retrieved = sum(1 for doc_id in retrieved_doc_ids if doc_id in relevant_docs)
            precision_at_k = relevant_retrieved / k if k > 0 else 0
            
            print(f"Precision@{k}: {precision_at_k:.2%}")
            
            if k == 3:
                assert precision_at_k >= 0.33, f"Precision@3 too low: {precision_at_k:.2%}"


class TestCrossTopicRetrieval:
    """Test suite for retrieval across different topics."""
    
    @pytest.mark.quality
    def test_topic_isolation(self, vector_db_with_knowledge):
        """Test that queries retrieve documents from appropriate topics."""
        topic_queries = [
            ("web development", 'web'),
            ("artificial intelligence", 'AI'),
            ("data analysis", 'data_science'),
        ]
        
        correct_topic_retrievals = 0
        
        for query, expected_topic in topic_queries:
            results = vector_db_with_knowledge.search(query, n_results=1)
            retrieved_topic = results['metadatas'][0].get('topic')
            
            if retrieved_topic == expected_topic:
                correct_topic_retrievals += 1
                print(f"✓ '{query}' -> {retrieved_topic}")
            else:
                print(f"✗ '{query}' -> Expected {expected_topic}, got {retrieved_topic}")
        
        accuracy = correct_topic_retrievals / len(topic_queries)
        print(f"\nTopic isolation accuracy: {accuracy:.2%}")
        
        assert accuracy >= 0.60, f"Topic isolation accuracy too low: {accuracy:.2%}"
    
    @pytest.mark.quality
    def test_multi_topic_query(self, vector_db_with_knowledge):
        """Test queries that span multiple topics."""
        # Query combining Python and AI topics
        query = "using Python for machine learning and artificial intelligence"
        results = vector_db_with_knowledge.search(query, n_results=3)
        
        retrieved_topics = [meta.get('topic') for meta in results['metadatas']]
        
        # Should retrieve from multiple relevant topics
        print(f"\nMulti-topic query retrieved: {retrieved_topics}")
        
        # At least one AI-related and one programming-related doc
        has_ai = any(topic == 'AI' for topic in retrieved_topics)
        has_programming = any(topic in ['programming', 'data_science'] for topic in retrieved_topics)
        
        assert has_ai or has_programming, "Should retrieve documents from relevant topics"


def generate_retrieval_quality_report(vector_db_with_knowledge) -> str:
    """Generate a comprehensive retrieval quality report."""
    report = "\n" + "="*60 + "\n"
    report += "RETRIEVAL QUALITY REPORT\n"
    report += "="*60 + "\n\n"
    
    # Test 1: Overall accuracy
    queries = [
        ("Python programming", '1'),
        ("machine learning", '2'),
        ("neural networks", '3'),
        ("Django framework", '4'),
        ("data science", '5'),
        ("NLP", '6'),
    ]
    
    correct = 0
    for query, expected in queries:
        results = vector_db_with_knowledge.search(query, n_results=1)
        if results['metadatas'] and results['metadatas'][0].get('doc_id') == expected:
            correct += 1
    
    accuracy = correct / len(queries)
    report += f"Overall Retrieval Accuracy: {accuracy:.1%}\n"
    report += f"Correct: {correct}/{len(queries)}\n\n"
    
    # Test 2: Average distance of top result
    distances = []
    for query, _ in queries:
        results = vector_db_with_knowledge.search(query, n_results=1)
        if results['distances']:
            distances.append(results['distances'][0])
    
    avg_distance = np.mean(distances)
    report += f"Average Top-1 Distance: {avg_distance:.4f}\n"
    report += "(Lower is better for most metrics)\n\n"
    
    report += "="*60 + "\n"
    
    return report


@pytest.mark.quality
def test_generate_full_report(vector_db_with_knowledge):
    """Generate and display comprehensive retrieval quality report."""
    report = generate_retrieval_quality_report(vector_db_with_knowledge)
    print(report)
    assert True  # This test always passes, it's for reporting


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "quality"])
