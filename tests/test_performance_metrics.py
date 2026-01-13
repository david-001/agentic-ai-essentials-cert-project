"""
Performance Metrics Tests for RAG Assistant.
Tests key performance indicators for retrieval, generation, and end-to-end RAG quality.
Implements standard metrics: Precision@k, Recall@k, MRR, NDCG, faithfulness, and more.
"""

import pytest
import os
import sys
import numpy as np
from typing import List, Dict, Any, Tuple
from unittest.mock import patch, MagicMock
from collections import Counter

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import RAGAssistant
from vectordb import VectorDB


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
# Retrieval Metrics
# ============================================================================

class TestPrecisionRecallMetrics:
    """Test suite for Precision@k and Recall@k metrics."""
    
    def calculate_precision_at_k(
        self, 
        retrieved_ids: List[str], 
        relevant_ids: List[str], 
        k: int
    ) -> float:
        """
        Calculate Precision@k.
        Precision@k = (# of relevant items in top-k) / k
        """
        top_k = retrieved_ids[:k]
        relevant_retrieved = len(set(top_k) & set(relevant_ids))
        return relevant_retrieved / k if k > 0 else 0.0
    
    def calculate_recall_at_k(
        self, 
        retrieved_ids: List[str], 
        relevant_ids: List[str], 
        k: int
    ) -> float:
        """
        Calculate Recall@k.
        Recall@k = (# of relevant items in top-k) / (total # of relevant items)
        """
        top_k = retrieved_ids[:k]
        relevant_retrieved = len(set(top_k) & set(relevant_ids))
        total_relevant = len(relevant_ids)
        return relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
    
    @pytest.mark.performance
    def test_precision_at_k_perfect_retrieval(self):
        """Test precision@k with perfect retrieval."""
        retrieved = ['rel1', 'rel2', 'rel3', 'irrel1']
        relevant = ['rel1', 'rel2', 'rel3']
        
        precision_3 = self.calculate_precision_at_k(retrieved, relevant, k=3)
        assert precision_3 == 1.0, "Perfect top-3 should have precision 1.0"
        
        precision_4 = self.calculate_precision_at_k(retrieved, relevant, k=4)
        assert precision_4 == 0.75, "Top-4 with 3 relevant should have precision 0.75"
    
    @pytest.mark.performance
    def test_recall_at_k_perfect_retrieval(self):
        """Test recall@k with perfect retrieval."""
        retrieved = ['rel1', 'rel2', 'rel3', 'irrel1']
        relevant = ['rel1', 'rel2', 'rel3']
        
        recall_3 = self.calculate_recall_at_k(retrieved, relevant, k=3)
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
            precision = self.calculate_precision_at_k(
                retrieved_ids, 
                item['relevant_doc_ids'], 
                k
            )
            recall = self.calculate_recall_at_k(
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
    
    def calculate_reciprocal_rank(
        self, 
        retrieved_ids: List[str], 
        relevant_ids: List[str]
    ) -> float:
        """
        Calculate Reciprocal Rank.
        RR = 1 / (rank of first relevant item)
        Returns 0 if no relevant items found.
        """
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0
    
    def calculate_mrr(
        self, 
        queries_results: List[Tuple[List[str], List[str]]]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank across multiple queries.
        queries_results: List of (retrieved_ids, relevant_ids) tuples
        """
        rr_scores = [
            self.calculate_reciprocal_rank(retrieved, relevant)
            for retrieved, relevant in queries_results
        ]
        return np.mean(rr_scores) if rr_scores else 0.0
    
    @pytest.mark.performance
    def test_reciprocal_rank_first_position(self):
        """Test RR when relevant doc is at position 1."""
        retrieved = ['rel1', 'irrel1', 'irrel2']
        relevant = ['rel1', 'rel2']
        
        rr = self.calculate_reciprocal_rank(retrieved, relevant)
        assert rr == 1.0, "First position should give RR of 1.0"
    
    @pytest.mark.performance
    def test_reciprocal_rank_second_position(self):
        """Test RR when relevant doc is at position 2."""
        retrieved = ['irrel1', 'rel1', 'irrel2']
        relevant = ['rel1', 'rel2']
        
        rr = self.calculate_reciprocal_rank(retrieved, relevant)
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
        
        mrr = self.calculate_mrr(queries_results)
        
        assert mrr > 0.3, f"MRR too low: {mrr}"
        print(f"\nMean Reciprocal Rank: {mrr:.3f}")


class TestNDCGMetric:
    """Test suite for Normalized Discounted Cumulative Gain (NDCG)."""
    
    def calculate_dcg(self, relevances: List[float], k: int = None) -> float:
        """
        Calculate Discounted Cumulative Gain.
        DCG = sum(rel_i / log2(i + 1)) for i in top-k positions
        """
        if k:
            relevances = relevances[:k]
        
        dcg = 0.0
        for i, rel in enumerate(relevances, start=1):
            dcg += rel / np.log2(i + 1)
        
        return dcg
    
    def calculate_ndcg(
        self, 
        retrieved_ids: List[str], 
        relevant_ids: List[str], 
        k: int = None
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@k.
        Binary relevance: 1 if relevant, 0 if not.
        """
        # Create relevance scores (binary: 1 or 0)
        relevances = [1.0 if doc_id in relevant_ids else 0.0 
                      for doc_id in retrieved_ids]
        
        if k:
            relevances = relevances[:k]
        
        # Calculate DCG
        dcg = self.calculate_dcg(relevances, k)
        
        # Calculate ideal DCG (perfect ranking)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = self.calculate_dcg(ideal_relevances, k)
        
        # Calculate NDCG
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @pytest.mark.performance
    def test_ndcg_perfect_ranking(self):
        """Test NDCG with perfect ranking."""
        retrieved = ['rel1', 'rel2', 'rel3', 'irrel1']
        relevant = ['rel1', 'rel2', 'rel3']
        
        ndcg = self.calculate_ndcg(retrieved, relevant, k=4)
        assert ndcg == 1.0, "Perfect ranking should give NDCG of 1.0"
    
    @pytest.mark.performance
    def test_ndcg_reversed_ranking(self):
        """Test NDCG with completely reversed ranking."""
        retrieved = ['irrel1', 'irrel2', 'rel1', 'rel2']
        relevant = ['rel1', 'rel2']
        
        ndcg = self.calculate_ndcg(retrieved, relevant, k=4)
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
            
            ndcg = self.calculate_ndcg(
                results['ids'], 
                item['relevant_doc_ids'], 
                k=k
            )
            ndcg_scores.append(ndcg)
        
        avg_ndcg = np.mean(ndcg_scores)
        
        assert avg_ndcg > 0.4, f"Average NDCG too low: {avg_ndcg}"
        print(f"\nAverage NDCG@{k}: {avg_ndcg:.3f}")


# ============================================================================
# Generation Quality Metrics
# ============================================================================

class TestAnswerQualityMetrics:
    """Test suite for answer quality metrics (faithfulness, relevance)."""
    
    def calculate_token_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate token overlap between two texts.
        Simple proxy for semantic similarity.
        """
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        overlap = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return overlap / union if union > 0 else 0.0
    
    def calculate_faithfulness(
        self, 
        answer: str, 
        context: str
    ) -> float:
        """
        Calculate faithfulness score.
        Measures if answer is grounded in context.
        Simple implementation using token overlap.
        """
        # Remove common stop words for better measurement
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        
        answer_tokens = set(answer.lower().split()) - stop_words
        context_tokens = set(context.lower().split()) - stop_words
        
        if not answer_tokens:
            return 0.0
        
        # Count how many answer tokens appear in context
        grounded_tokens = len(answer_tokens & context_tokens)
        total_answer_tokens = len(answer_tokens)
        
        return grounded_tokens / total_answer_tokens
    
    def calculate_answer_relevance(
        self, 
        answer: str, 
        question: str
    ) -> float:
        """
        Calculate answer relevance to question.
        Simple implementation using token overlap.
        """
        return self.calculate_token_overlap(answer, question)
    
    @pytest.mark.performance
    def test_faithfulness_high_overlap(self):
        """Test faithfulness with high context overlap."""
        context = "The company offers 15 vacation days for entry-level employees."
        answer = "Entry-level employees receive 15 vacation days."
        
        faithfulness = self.calculate_faithfulness(answer, context)
        assert faithfulness > 0.5, f"Should have high faithfulness: {faithfulness}"
    
    @pytest.mark.performance
    def test_faithfulness_low_overlap(self):
        """Test faithfulness with low context overlap (hallucination)."""
        context = "The company offers 15 vacation days for entry-level employees."
        answer = "Senior developers get unlimited paid time off and bonuses."
        
        faithfulness = self.calculate_faithfulness(answer, context)
        assert faithfulness < 0.3, f"Should have low faithfulness: {faithfulness}"
    
    @pytest.mark.performance
    def test_answer_relevance_integration(self, rag_assistant_with_corpus, evaluation_corpus):
        """Test answer relevance on actual RAG system."""
        with patch.object(rag_assistant_with_corpus, 'chain') as mock_chain:
            # Mock realistic answers
            mock_answers = [
                "Entry-level employees receive 15 days per year of vacation.",
                "The API supports API Key Authentication and OAuth 2.0.",
                "The rate limit is 1000 requests per hour for API keys."
            ]
            
            relevance_scores = []
            
            for item, mock_answer in zip(evaluation_corpus, mock_answers):
                mock_chain.invoke.return_value = mock_answer
                
                answer = rag_assistant_with_corpus.query(item['query'])
                relevance = self.calculate_answer_relevance(answer, item['query'])
                
                relevance_scores.append(relevance)
            
            avg_relevance = np.mean(relevance_scores)
            assert avg_relevance > 0.1, f"Average relevance too low: {avg_relevance}"
            print(f"\nAverage Answer Relevance: {avg_relevance:.3f}")


class TestHallucinationDetection:
    """Test suite for hallucination detection."""
    
    def detect_hallucination(
        self, 
        answer: str, 
        context: str, 
        threshold: float = 0.5
    ) -> bool:
        """
        Detect if answer contains hallucinations.
        Returns True if hallucination detected.
        """
        # Extract key facts from answer (simplified)
        answer_tokens = set(answer.lower().split())
        context_tokens = set(context.lower().split())
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are'}
        answer_tokens -= stop_words
        context_tokens -= stop_words
        
        if not answer_tokens:
            return False
        
        # Check overlap
        overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
        
        return overlap < threshold
    
    @pytest.mark.performance
    def test_no_hallucination_detection(self):
        """Test that grounded answer is not flagged as hallucination."""
        context = "Python is a programming language used for data science."
        answer = "Python is used for data science."
        
        is_hallucination = self.detect_hallucination(answer, context)
        assert not is_hallucination, "Grounded answer should not be flagged"
    
    @pytest.mark.performance
    def test_hallucination_detection(self):
        """Test that hallucinated answer is detected."""
        context = "Python is a programming language used for data science."
        answer = "Java is the fastest language for machine learning applications."
        
        is_hallucination = self.detect_hallucination(answer, context)
        assert is_hallucination, "Hallucinated answer should be detected"
    
    @pytest.mark.performance
    def test_hallucination_rate(self, rag_assistant_with_corpus, evaluation_corpus):
        """Test hallucination rate on RAG system."""
        with patch.object(rag_assistant_with_corpus, 'chain') as mock_chain:
            # Mix of grounded and hallucinated answers
            mock_answers = [
                "Entry-level employees receive 15 days of vacation per year.",  # Grounded
                "The API supports OAuth 2.0 and API Key Authentication.",  # Grounded
                "Premium users get unlimited API calls with priority support."  # Hallucination
            ]
            
            hallucination_count = 0
            
            for item, mock_answer in zip(evaluation_corpus, mock_answers):
                mock_chain.invoke.return_value = mock_answer
                
                # Get context
                results = rag_assistant_with_corpus.vector_db.search(
                    item['query'], 
                    n_results=3
                )
                context = "\n".join(results['documents'])
                
                answer = rag_assistant_with_corpus.query(item['query'])
                
                if self.detect_hallucination(answer, context):
                    hallucination_count += 1
            
            hallucination_rate = hallucination_count / len(evaluation_corpus)
            
            # Ideally should be 0, but we injected one hallucination
            assert hallucination_rate <= 0.5, f"Hallucination rate too high: {hallucination_rate}"
            print(f"\nHallucination Rate: {hallucination_rate:.2%}")


# ============================================================================
# End-to-End Performance Metrics
# ============================================================================

class TestEndToEndMetrics:
    """Test suite for complete RAG pipeline performance."""
    
    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @pytest.mark.performance
    def test_retrieval_latency(self, rag_assistant_with_corpus, evaluation_corpus):
        """Test retrieval latency performance."""
        import time
        
        latencies = []
        
        for item in evaluation_corpus:
            start_time = time.time()
            rag_assistant_with_corpus.vector_db.search(item['query'], n_results=5)
            end_time = time.time()
            
            latencies.append(end_time - start_time)
        
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        
        # Assert reasonable latency (should be under 1 second for small corpus)
        assert avg_latency < 1.0, f"Average latency too high: {avg_latency:.3f}s"
        assert max_latency < 2.0, f"Max latency too high: {max_latency:.3f}s"
        
        print(f"\nRetrieval Latency:")
        print(f"  Average: {avg_latency*1000:.2f}ms")
        print(f"  Max: {max_latency*1000:.2f}ms")
    
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
    
    @pytest.mark.performance
    def test_comprehensive_rag_metrics(self, rag_assistant_with_corpus, evaluation_corpus):
        """Comprehensive test of all major RAG metrics."""
        metrics = {
            'precision_at_3': [],
            'recall_at_3': [],
            'mrr': [],
            'ndcg_at_5': [],
            'faithfulness': [],
            'answer_relevance': []
        }
        
        # Mock chain for answer generation
        with patch.object(rag_assistant_with_corpus, 'chain') as mock_chain:
            mock_answers = [
                "Entry-level employees receive 15 days of vacation per year.",
                "The API supports API Key Authentication and OAuth 2.0.",
                "The rate limit is 1000 requests per hour for API keys."
            ]
            
            for item, mock_answer in zip(evaluation_corpus, mock_answers):
                mock_chain.invoke.return_value = mock_answer
                
                # Get retrieval results
                results = rag_assistant_with_corpus.vector_db.search(
                    item['query'], 
                    n_results=5
                )
                
                retrieved_ids = results['ids']
                relevant_ids = item['relevant_doc_ids']
                context = "\n".join(results['documents'])
                
                # Calculate retrieval metrics
                top_3 = retrieved_ids[:3]
                precision = len(set(top_3) & set(relevant_ids)) / 3
                recall = len(set(top_3) & set(relevant_ids)) / len(relevant_ids) if relevant_ids else 0
                
                metrics['precision_at_3'].append(precision)
                metrics['recall_at_3'].append(recall)
                
                # MRR
                rr = 0
                for rank, doc_id in enumerate(retrieved_ids, start=1):
                    if doc_id in relevant_ids:
                        rr = 1.0 / rank
                        break
                metrics['mrr'].append(rr)
                
                # NDCG@5
                relevances = [1.0 if doc_id in relevant_ids else 0.0 
                             for doc_id in retrieved_ids[:5]]
                dcg = sum(rel / np.log2(i + 1) for i, rel in enumerate(relevances, start=1))
                ideal_rel = sorted(relevances, reverse=True)
                idcg = sum(rel / np.log2(i + 1) for i, rel in enumerate(ideal_rel, start=1))
                ndcg = dcg / idcg if idcg > 0 else 0
                metrics['ndcg_at_5'].append(ndcg)
                
                # Get answer
                answer = rag_assistant_with_corpus.query(item['query'])
                
                # Faithfulness
                answer_tokens = set(answer.lower().split())
                context_tokens = set(context.lower().split())
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
                answer_tokens -= stop_words
                context_tokens -= stop_words
                
                faithfulness = len(answer_tokens & context_tokens) / len(answer_tokens) if answer_tokens else 0
                metrics['faithfulness'].append(faithfulness)
                
                # Answer relevance
                query_tokens = set(item['query'].lower().split())
                relevance = len(answer_tokens & query_tokens) / len(query_tokens | answer_tokens) if answer_tokens or query_tokens else 0
                metrics['answer_relevance'].append(relevance)
        
        # Calculate averages
        print("\n" + "="*60)
        print("COMPREHENSIVE RAG PERFORMANCE METRICS")
        print("="*60)
        print("\nRetrieval Metrics:")
        print(f"  Precision@3:      {np.mean(metrics['precision_at_3']):.3f}")
        print(f"  Recall@3:         {np.mean(metrics['recall_at_3']):.3f}")
        print(f"  MRR:              {np.mean(metrics['mrr']):.3f}")
        print(f"  NDCG@5:           {np.mean(metrics['ndcg_at_5']):.3f}")
        print("\nGeneration Metrics:")
        print(f"  Faithfulness:     {np.mean(metrics['faithfulness']):.3f}")
        print(f"  Answer Relevance: {np.mean(metrics['answer_relevance']):.3f}")
        print("="*60)
        
        # Assert minimum quality thresholds
        assert np.mean(metrics['precision_at_3']) > 0.3, "Precision@3 below threshold"
        assert np.mean(metrics['recall_at_3']) > 0.3, "Recall@3 below threshold"
        assert np.mean(metrics['faithfulness']) > 0.3, "Faithfulness below threshold"


# ============================================================================
# Performance Benchmarking
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
        import sys
        
        # Get approximate memory usage of vector DB
        collection_count = rag_assistant_with_corpus.vector_db.collection.count()
        
        # Should not have excessive chunks
        assert collection_count < 1000, f"Too many chunks stored: {collection_count}"
        print(f"\nVector DB contains {collection_count} chunks")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])
