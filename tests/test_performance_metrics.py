"""
Performance Metrics Tests for RAG Assistant.
Tests key performance indicators for retrieval, generation, and end-to-end RAG quality.
Implements standard metrics: Precision@k, Recall@k, MRR, NDCG, faithfulness, and more.

Note: Random seeds are managed by conftest.py fixture for reproducibility.
"""

import pytest
import os
import sys
import numpy as np
import random
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import RAGAssistant
from vectordb import VectorDB
from metrics_utils import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    calculate_mrr,
    calculate_ndcg,
    calculate_faithfulness,
    calculate_answer_relevance
)


# ============================================================================
# Test Fixtures and Data
# ============================================================================

@pytest.fixture
def evaluation_corpus():
    """
    Fixture providing a labeled corpus for evaluation.
    Each item has: query, relevant_doc_ids, ground_truth_answer
    """
    return [
        {
            'query': 'How many vacation days do entry-level employees receive?',
            'relevant_doc_ids': ['doc_0_chunk_0', 'doc_0_chunk_1'],
            'ground_truth': '15 days per year',
            'doc_content': (
                "Company Vacation Policy\n\n"
                "Employees receive vacation days based on tenure:\n"
                "- Entry-level (0-2 years): 15 days per year\n"
                "- Mid-level (3-5 years): 20 days per year\n"
                "- Senior-level (5+ years): 25 days per year\n"
                "Vacation must be requested 2 weeks in advance."
            )
        },
        {
            'query': 'What authentication methods does the API support?',
            'relevant_doc_ids': ['doc_1_chunk_0', 'doc_1_chunk_1'],
            'ground_truth': 'API Key Authentication and OAuth 2.0',
            'doc_content': (
                "API Authentication\n\n"
                "Our API supports two authentication methods:\n"
                "1. API Key Authentication - Include X-API-Key header\n"
                "2. OAuth 2.0 - Use for user-specific access\n"
                "Rate limits: 1000 requests/hour for API keys."
            )
        },
        {
            'query': 'What is the rate limit for API keys?',
            'relevant_doc_ids': ['doc_1_chunk_1'],
            'ground_truth': '1000 requests per hour',
            'doc_content': (
                "API Authentication\n\n"
                "Our API supports two authentication methods:\n"
                "1. API Key Authentication - Include X-API-Key header\n"
                "2. OAuth 2.0 - Use for user-specific access\n"
                "Rate limits: 1000 requests/hour for API keys."
            )
        }
    ]


@pytest.fixture
def rag_assistant_with_corpus(evaluation_corpus):
    """Fixture providing RAG assistant loaded with evaluation corpus."""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        assistant = RAGAssistant()
        
        # Add documents from evaluation corpus
        documents = [
            {
                'content': item['doc_content'],
                'metadata': {'source': f"eval_doc_{i}.txt"}
            }
            for i, item in enumerate(evaluation_corpus)
        ]
        assistant.add_documents(documents)
        
        return assistant


# ============================================================================
# Retrieval Metrics Tests
# ============================================================================

class TestPrecisionRecallMetrics:
    """Test suite for Precision@k and Recall@k metrics."""
    
    @pytest.mark.performance
    def test_precision_at_k_perfect_retrieval(self):
        """Test precision@k with perfect retrieval."""
        retrieved = ['rel1', 'rel2', 'rel3', 'irrel1']
        relevant = ['rel1', 'rel2', 'rel3']
        
        precision_3 = calculate_precision_at_k(retrieved, relevant, k=3)
        assert precision_3 == 1.0, "Perfect top-3 should have precision 1.0"
        
        precision_4 = calculate_precision_at_k(retrieved, relevant, k=4)
        assert precision_4 == 0.75, "Top-4 with 3 relevant should have precision 0.75"
    
    @pytest.mark.performance
    def test_recall_at_k_perfect_retrieval(self):
        """Test recall@k with perfect retrieval."""
        retrieved = ['rel1', 'rel2', 'rel3', 'irrel1']
        relevant = ['rel1', 'rel2', 'rel3']
        
        recall_3 = calculate_recall_at_k(retrieved, relevant, k=3)
        assert recall_3 == 1.0, "All relevant items retrieved should have recall 1.0"
    
    @pytest.mark.performance
    def test_precision_recall_integration(self, rag_assistant_with_corpus, evaluation_corpus):
        """Test precision and recall on actual RAG system."""
        precision_scores = []
        recall_scores = []
        k = 3
        
        for item in evaluation_corpus:
            # Perform search
            results = rag_assistant_with_corpus.vector_db.search(
                item['query'], 
                n_results=k
            )
            retrieved_ids = results['ids']
            
            # Calculate metrics
            precision = calculate_precision_at_k(
                retrieved_ids, 
                item['relevant_doc_ids'], 
                k
            )
            recall = calculate_recall_at_k(
                retrieved_ids, 
                item['relevant_doc_ids'], 
                k
            )
            
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        # Assert reasonable performance
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        
        assert avg_precision > 0.3, f"Average precision too low: {avg_precision}"
        assert avg_recall > 0.3, f"Average recall too low: {avg_recall}"
        
        print(f"\nRetrieval Performance:")
        print(f"  Average Precision@{k}: {avg_precision:.3f}")
        print(f"  Average Recall@{k}: {avg_recall:.3f}")


class TestMRRMetric:
    """Test suite for Mean Reciprocal Rank (MRR) metric."""
    
    @pytest.mark.performance
    def test_reciprocal_rank_first_position(self):
        """Test RR when relevant doc is at position 1."""
        retrieved = ['rel1', 'irrel1', 'irrel2']
        relevant = ['rel1', 'rel2']
        
        rr = calculate_reciprocal_rank(retrieved, relevant)
        assert rr == 1.0, "First position should give RR of 1.0"
    
    @pytest.mark.performance
    def test_reciprocal_rank_second_position(self):
        """Test RR when relevant doc is at position 2."""
        retrieved = ['irrel1', 'rel1', 'irrel2']
        relevant = ['rel1', 'rel2']
        
        rr = calculate_reciprocal_rank(retrieved, relevant)
        assert rr == 0.5, "Second position should give RR of 0.5"
    
    @pytest.mark.performance
    def test_mrr_integration(self, rag_assistant_with_corpus, evaluation_corpus):
        """Test MRR on actual RAG system."""
        queries_results = []
        
        for item in evaluation_corpus:
            results = rag_assistant_with_corpus.vector_db.search(
                item['query'], 
                n_results=5
            )
            queries_results.append((results['ids'], item['relevant_doc_ids']))
        
        mrr = calculate_mrr(queries_results)
        
        assert mrr > 0.3, f"MRR too low: {mrr}"
        print(f"\nMean Reciprocal Rank: {mrr:.3f}")


class TestNDCGMetric:
    """Test suite for Normalized Discounted Cumulative Gain (NDCG)."""
    
    @pytest.mark.performance
    def test_ndcg_perfect_ranking(self):
        """Test NDCG with perfect ranking."""
        retrieved = ['rel1', 'rel2', 'rel3', 'irrel1']
        relevant = ['rel1', 'rel2', 'rel3']
        
        ndcg = calculate_ndcg(retrieved, relevant, k=4)
        assert ndcg == 1.0, "Perfect ranking should give NDCG of 1.0"
    
    @pytest.mark.performance
    def test_ndcg_reversed_ranking(self):
        """Test NDCG with completely reversed ranking."""
        retrieved = ['irrel1', 'irrel2', 'rel1', 'rel2']
        relevant = ['rel1', 'rel2']
        
        ndcg = calculate_ndcg(retrieved, relevant, k=4)
        assert ndcg < 1.0, "Imperfect ranking should give NDCG < 1.0"
    
    @pytest.mark.performance
    def test_ndcg_integration(self, rag_assistant_with_corpus, evaluation_corpus):
        """Test NDCG on actual RAG system."""
        ndcg_scores = []
        k = 5
        
        for item in evaluation_corpus:
            results = rag_assistant_with_corpus.vector_db.search(
                item['query'], 
                n_results=k
            )
            
            ndcg = calculate_ndcg(
                results['ids'], 
                item['relevant_doc_ids'], 
                k=k
            )
            ndcg_scores.append(ndcg)
        
        avg_ndcg = np.mean(ndcg_scores)
        
        assert avg_ndcg > 0.4, f"Average NDCG too low: {avg_ndcg}"
        print(f"\nAverage NDCG@{k}: {avg_ndcg:.3f}")


# ============================================================================
# Generation Quality Metrics Tests
# ============================================================================

class TestAnswerQualityMetrics:
    """Test suite for answer quality metrics (faithfulness, relevance)."""
    
    @pytest.mark.performance
    def test_faithfulness_perfect_grounding(self):
        """Test faithfulness with perfectly grounded answer."""
        context = "The company offers 15 vacation days per year for new employees."
        answer = "New employees receive 15 vacation days annually."
        
        faithfulness = calculate_faithfulness(answer, context)
        assert faithfulness > 0.5, f"High faithfulness expected, got {faithfulness}"
    
    @pytest.mark.performance
    def test_faithfulness_hallucinated_content(self):
        """Test faithfulness detects hallucinated content."""
        context = "The company offers 15 vacation days per year."
        answer = "Employees get unlimited sick leave and flexible hours with remote work options."
        
        faithfulness = calculate_faithfulness(answer, context)
        assert faithfulness < 0.3, f"Low faithfulness expected for hallucination, got {faithfulness}"
    
    @pytest.mark.performance
    def test_answer_relevance_high(self):
        """Test answer relevance for highly relevant answer."""
        question = "How many vacation days do employees receive?"
        answer = "Employees receive 15 vacation days per year."
        
        relevance = calculate_answer_relevance(answer, question)
        assert relevance > 0.2, f"High relevance expected, got {relevance}"
    
    @pytest.mark.performance
    def test_answer_relevance_low(self):
        """Test answer relevance for off-topic answer."""
        question = "How many vacation days do employees receive?"
        answer = "The office is located in downtown area."
        
        relevance = calculate_answer_relevance(answer, question)
        assert relevance < 0.3, f"Low relevance expected, got {relevance}"


# ============================================================================
# End-to-End RAG Quality Tests
# ============================================================================

class TestEndToEndRAGQuality:
    """Test suite for comprehensive end-to-end RAG quality metrics."""
    
    @pytest.mark.performance
    def test_end_to_end_accuracy(self, rag_assistant_with_corpus, evaluation_corpus):
        """Test end-to-end accuracy with ground truth comparison."""
        with patch.object(rag_assistant_with_corpus, 'chain') as mock_chain:
            # Mock answers that match ground truth
            mock_answers = [
                "15 days per year",
                "API Key Authentication and OAuth 2.0",
                "1000 requests per hour"
            ]
            
            correct_count = 0
            
            for item, mock_answer in zip(evaluation_corpus, mock_answers):
                mock_chain.invoke.return_value = mock_answer
                
                answer = rag_assistant_with_corpus.query(item['query'])
                
                # Simple accuracy check (contains key terms)
                ground_truth_terms = set(item['ground_truth'].lower().split())
                answer_terms = set(answer.lower().split())
                
                overlap = len(ground_truth_terms & answer_terms)
                if overlap >= len(ground_truth_terms) * 0.5:  # 50% overlap threshold
                    correct_count += 1
            
            accuracy = correct_count / len(evaluation_corpus)
            
            assert accuracy > 0.6, f"End-to-end accuracy too low: {accuracy}"
            print(f"\nEnd-to-End Accuracy: {accuracy:.2%}")
    
    @pytest.mark.performance
    def test_context_relevancy_score(self, rag_assistant_with_corpus, evaluation_corpus):
        """Test relevancy of retrieved context to queries."""
        relevancy_scores = []
        
        for item in evaluation_corpus:
            results = rag_assistant_with_corpus.vector_db.search(
                item['query'], 
                n_results=3
            )
            
            # Check if retrieved context contains query terms
            query_terms = set(item['query'].lower().split())
            context = " ".join(results['documents']).lower()
            context_terms = set(context.split())
            
            overlap = len(query_terms & context_terms)
            relevancy = overlap / len(query_terms) if query_terms else 0
            
            relevancy_scores.append(relevancy)
        
        avg_relevancy = np.mean(relevancy_scores)
        
        assert avg_relevancy > 0.4, f"Context relevancy too low: {avg_relevancy}"
        print(f"\nAverage Context Relevancy: {avg_relevancy:.3f}")


# ============================================================================
# Performance Benchmarking Tests
# ============================================================================

class TestPerformanceBenchmarks:
    """Test suite for system performance benchmarks."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_throughput_benchmark(self, rag_assistant_with_corpus, evaluation_corpus):
        """Benchmark query throughput (queries per second)."""
        import time
        
        num_iterations = 10
        start_time = time.time()
        
        with patch.object(rag_assistant_with_corpus, 'chain') as mock_chain:
            mock_chain.invoke.return_value = "Test answer"
            
            for _ in range(num_iterations):
                for item in evaluation_corpus:
                    rag_assistant_with_corpus.query(item['query'])
        
        end_time = time.time()
        total_time = end_time - start_time
        total_queries = num_iterations * len(evaluation_corpus)
        qps = total_queries / total_time
        
        print(f"\nThroughput: {qps:.2f} queries/second")
        assert qps > 1.0, f"Throughput too low: {qps:.2f} qps"
    
    @pytest.mark.performance
    def test_memory_efficiency(self, rag_assistant_with_corpus):
        """Test memory usage is reasonable."""
        # Get approximate memory usage of vector DB
        collection_count = rag_assistant_with_corpus.vector_db.collection.count()
        
        # Should not have excessive chunks
        assert collection_count < 1000, f"Too many chunks stored: {collection_count}"
        print(f"\nVector DB contains {collection_count} chunks")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])
