"""
# Evaluates RAG system quality by measuring retrieval accuracy (Precision, Recall, MRR, NDCG) and generation quality (Faithfulness, Relevance) using DeepEval
# Outputs a graded console report.
"""

import os
import sys
import numpy as np
import time
import random
from datetime import datetime
from typing import Dict, List, Any

# DeepEval imports
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase
from deepeval.models import GPTModel, GeminiModel

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from metrics_utils import (
    calculate_precision_at_k, 
    calculate_recall_at_k, 
    calculate_mrr, 
    calculate_ndcg
)
from rag_evaluator_utils import get_status
from app import RAGAssistant

# Set random seeds for reproducibility when run standalone
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class RagEvaluator:
    """Generate and export performance metrics reports using DeepEval."""
    
    def __init__(self):
        """Initialize the performance reporter."""
        self.metrics = {}
        self.timestamp = datetime.now().isoformat()
        self.model = self._initialize_deepeval_model()
        self.rag_assistant = None
        
        # Initialize DeepEval metrics
        self._init_deepeval_metrics()

    def _init_deepeval_metrics(self):
        """Initialize all DeepEval metric instances."""
        self.faithfulness_metric = FaithfulnessMetric(
            threshold=0.7, model=self.model, include_reason=True
        )
        self.answer_relevancy_metric = AnswerRelevancyMetric(
            threshold=0.5, model=self.model, include_reason=True
        )
        self.contextual_precision_metric = ContextualPrecisionMetric(
            threshold=0.5, model=self.model, include_reason=True
        )
        self.contextual_recall_metric = ContextualRecallMetric(
            threshold=0.5, model=self.model, include_reason=True
        )
        self.contextual_relevancy_metric = ContextualRelevancyMetric(
            threshold=0.5, model=self.model, include_reason=True
        )

    def _initialize_deepeval_model(self):
        """Initialize the LLM model for DeepEval metrics."""
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"✓ Using OpenAI: {model_name}")
            self.model_provider = 'openai'
            return GPTModel(model=model_name, api_key=os.getenv("OPENAI_API_KEY"))
        
        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"✓ Using Groq: {model_name}")
            self.model_provider = 'groq'
            return GPTModel(
                model=model_name,
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1"
            )
        
        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"✓ Using Gemini: {model_name}")
            self.model_provider = 'gemini'
            return GeminiModel(model=model_name, api_key=os.getenv("GOOGLE_API_KEY"))
        
        else:
            raise ValueError(
                "No API key found. Set: OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY"
            )
    
    def initialize_rag_system(self) -> RAGAssistant:
        """Initialize the RAG system with documents."""
        if not any([os.getenv(k) for k in ["OPENAI_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY"]]):
            raise ValueError("No API key found. Set: OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY")
        
        self.rag_assistant = RAGAssistant()
        return self.rag_assistant
    
    def evaluate_rag_system(
        self,
        test_queries: List[Dict[str, Any]],
        n_results: int = 3
    ) -> Dict[str, float]:
        """Evaluate the RAG system with test queries."""
        if self.rag_assistant is None:
            print("Initializing RAG system...")
            self.initialize_rag_system()
        
        retrieval_results = []
        generation_results = []
        
        print(f"\nEvaluating RAG system with {len(test_queries)} test queries...")
        
        for i, test_item in enumerate(test_queries):
            query = test_item['query']
            relevant_ids = test_item.get('relevant_doc_ids', [])
            ground_truth = test_item.get('ground_truth', '')
            
            print(f"  Query {i+1}/{len(test_queries)}: {query[:50]}...")
            
            # Measure retrieval
            start_time = time.time()
            search_results = self.rag_assistant.vector_db.search(query, n_results=n_results)
            retrieval_latency = time.time() - start_time
            
            retrieval_results.append({
                'retrieved_ids': search_results['ids'],
                'relevant_ids': relevant_ids,
                'query': query,
                'latency': retrieval_latency
            })
            
            # Generate answer
            try:
                answer = self.rag_assistant.query(query, n_results=n_results)
            except Exception as e:
                print(f"    Warning: Answer generation failed: {e}")
                answer = f"Error: {str(e)}"
            
            generation_results.append({
                'answer': answer,
                'context': search_results['documents'],
                'query': query,
                'ground_truth': ground_truth
            })
        
        # Calculate metrics
        print("\nCalculating metrics...")
        self._calculate_retrieval_metrics(retrieval_results, n_results)
        self._calculate_generation_metrics(generation_results)
        
        return self.metrics
    
    def _calculate_retrieval_metrics(self, results: List[Dict], k: int = 3):
        """Calculate retrieval metrics."""
        precision_scores = []
        recall_scores = []
        reciprocal_ranks = []
        ndcg_scores = []
        latencies = []
        
        for result in results:
            retrieved = result['retrieved_ids']
            relevant = result['relevant_ids']
            
            if relevant:  # Only calculate if there are relevant docs
                precision_scores.append(calculate_precision_at_k(retrieved, relevant, k))
                recall_scores.append(calculate_recall_at_k(retrieved, relevant, k))
                reciprocal_ranks.append(calculate_mrr([(retrieved, relevant)]))
                ndcg_scores.append(calculate_ndcg(retrieved, relevant, k=5))
            
            latencies.append(result['latency'] * 1000)  # Convert to ms
        
        self.metrics['precision_at_3'] = np.mean(precision_scores) if precision_scores else 0
        self.metrics['recall_at_3'] = np.mean(recall_scores) if recall_scores else 0
        self.metrics['mrr'] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
        self.metrics['ndcg_at_5'] = np.mean(ndcg_scores) if ndcg_scores else 0
        self.metrics['avg_latency_ms'] = np.mean(latencies) if latencies else 0
    
    def _calculate_generation_metrics(self, results: List[Dict]):
        """Calculate generation metrics using DeepEval."""
        faithfulness_scores = []
        relevancy_scores = []
        precision_scores = []
        recall_scores = []
        relevancy_context_scores = []
        
        for result in results:
            if not result['context'] or not result['ground_truth']:
                continue
            
            test_case = LLMTestCase(
                input=result['query'],
                actual_output=result['answer'],
                expected_output=result['ground_truth'],
                retrieval_context=result['context']
            )
            
            try:
                # Faithfulness
                self.faithfulness_metric.measure(test_case)
                faithfulness_scores.append(self.faithfulness_metric.score)
                
                # Answer Relevancy
                self.answer_relevancy_metric.measure(test_case)
                relevancy_scores.append(self.answer_relevancy_metric.score)
                
                # Contextual Precision
                self.contextual_precision_metric.measure(test_case)
                precision_scores.append(self.contextual_precision_metric.score)
                
                # Contextual Recall
                self.contextual_recall_metric.measure(test_case)
                recall_scores.append(self.contextual_recall_metric.score)
                
                # Contextual Relevancy
                self.contextual_relevancy_metric.measure(test_case)
                relevancy_context_scores.append(self.contextual_relevancy_metric.score)
                
            except Exception as e:
                print(f"    Warning: Metric calculation failed: {e}")
        
        self.metrics['faithfulness'] = np.mean(faithfulness_scores) if faithfulness_scores else 0
        self.metrics['answer_relevance'] = np.mean(relevancy_scores) if relevancy_scores else 0
        self.metrics['contextual_precision'] = np.mean(precision_scores) if precision_scores else 0
        self.metrics['contextual_recall'] = np.mean(recall_scores) if recall_scores else 0
        self.metrics['contextual_relevancy'] = np.mean(relevancy_context_scores) if relevancy_context_scores else 0
    
    def _get_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get threshold values for each metric."""
        return {
            'precision_at_3': {'excellent': 0.8, 'good': 0.6},
            'recall_at_3': {'excellent': 0.8, 'good': 0.6},
            'mrr': {'excellent': 0.8, 'good': 0.7},
            'ndcg_at_5': {'excellent': 0.85, 'good': 0.7},
            'faithfulness': {'excellent': 0.8, 'good': 0.7},
            'answer_relevance': {'excellent': 0.7, 'good': 0.5},
            'avg_latency_ms': {'excellent': 200, 'good': 500},
            'contextual_precision': {'excellent': 0.8, 'good': 0.6},
            'contextual_recall': {'excellent': 0.8, 'good': 0.6},
            'contextual_relevancy': {'excellent': 0.8, 'good': 0.6}
        }
    
    def _calculate_grade(self) -> str:
        """Calculate overall grade."""
        if not self.metrics:
            return "No data"
        
        thresholds = self._get_thresholds()
        scores = []
        
        for metric, value in self.metrics.items():
            status = get_status(metric, value, thresholds)
            scores.append({'Excellent': 3, 'Good': 2, 'Needs Improvement': 1}.get(status, 0))
        
        if not scores:
            return "No data"
        
        avg_score = np.mean(scores)
        
        if avg_score >= 2.5:
            return "A (Excellent)"
        elif avg_score >= 2.0:
            return "B (Good)"
        elif avg_score >= 1.5:
            return "C (Acceptable)"
        else:
            return "D (Needs Improvement)"
    
    def print_console_report(self):
        """Print formatted report to console."""
        print("\n" + "="*70)
        print("RAG SYSTEM PERFORMANCE REPORT (DeepEval)")
        print("="*70)
        print(f"\nGenerated: {self.timestamp}")
        print(f"Model: {self.model}")
        
        if 'precision_at_3' in self.metrics:
            print("\n" + "-"*70)
            print("RETRIEVAL METRICS:")
            print(f"  Precision@3:  {self.metrics['precision_at_3']:.4f}")
            print(f"  Recall@3:     {self.metrics['recall_at_3']:.4f}")
            print(f"  MRR:          {self.metrics.get('mrr', 0):.4f}")
            print(f"  NDCG@5:       {self.metrics.get('ndcg_at_5', 0):.4f}")
            if 'avg_latency_ms' in self.metrics:
                print(f"  Avg Latency:  {self.metrics['avg_latency_ms']:.2f}ms")
        
        if 'faithfulness' in self.metrics:
            print("\nGENERATION METRICS (DeepEval):")
            print(f"  Faithfulness:          {self.metrics['faithfulness']:.4f}")
            print(f"  Answer Relevance:      {self.metrics['answer_relevance']:.4f}")
            print(f"  Contextual Precision:  {self.metrics.get('contextual_precision', 0):.4f}")
            print(f"  Contextual Recall:     {self.metrics.get('contextual_recall', 0):.4f}")
            print(f"  Contextual Relevancy:  {self.metrics.get('contextual_relevancy', 0):.4f}")
        
        print("\n" + "="*70)
        print(f"OVERALL GRADE: {self._calculate_grade()}")
        print("="*70)


# Example usage
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Direct RAG System Evaluation Example")
    print("-"*70)
    
    # Example test queries
    test_queries = [
        {
            'query': 'How many vacation days do entry-level employees receive?',
            'relevant_doc_ids': ['doc_2_chunk_33', 'doc_2_chunk_34'],
            'ground_truth': 'Entry-level employees receive 20 vacation days per year.'
        },
        {
            'query': 'What is the annual maximum benefit for dental?',
            'relevant_doc_ids': ['doc_2_chunk_26','doc_2_chunk_27'],
            'ground_truth': 'The annual maximum benefit for dental is $1,500.'
        },
        {
            'query': 'What is the rate limit for the Starter plan API keys?',
            'relevant_doc_ids': ['doc_0_chunk_11', 'doc_4_chunk_27'],
            'ground_truth': '100 requests per hour per API key'
        },
        {
            'query': 'Can I access the API without a key?',
            'relevant_doc_ids': ['doc_0_chunk_3', 'doc_0_chunk_5', 'doc_0_chunk_8'],
            'ground_truth': 'You cannot access the API without a key.'
        },
        {
            'query': 'What is the capital of France?',
            'relevant_doc_ids': [],
            'ground_truth': 'The question is not answerable given the documents.'
        },
        {
            'query': 'Are Visa credit cards accepted for payment?',
            'relevant_doc_ids': ['doc_1_chunk_16', 'doc_1_chunk_17'],
            'ground_truth': 'Yes, Visa credit cards are accepted as a payment method.'
        },
        {
            'query': 'How many therapy sessions are covered under mental health support?',
            'relevant_doc_ids': ['doc_2_chunk_29', 'doc_2_chunk_35'],
            'ground_truth': 'Unlimited therapy sessions are covered through the EAP (Employee Assistance Program).'
        },
        {
            'query': 'What are the core collaboration hours for employees?',
            'relevant_doc_ids': ['doc_2_chunk_8', 'doc_2_chunk_9'],
            'ground_truth': 'The core collaboration hours are 10:00 AM to 3:00 PM'
        },
        {
            'query': 'How long is the free trial period?',
            'relevant_doc_ids': ['doc_1_chunk_3', 'doc_1_chunk_4', 'doc_1_chunk_5'],
            'ground_truth': 'The free trial period is 14 days.'
        },
        {
            'query': 'Is there a mobile app?',
            'relevant_doc_ids': ['doc_1_chunk_9', 'doc_1_chunk_10', 'doc_1_chunk_11'],
            'ground_truth': 'Yes, there is a mobile app available.'
        },
        {
            'query': 'Can I get a refund?',
            'relevant_doc_ids': ['doc_1_chunk_30', 'doc_1_chunk_31', 'doc_1_chunk_32'],
            'ground_truth': 'Yes, you can get a refund on all annual subscriptions if you contact support within the first 30 days.'
        },
        {
            'query': 'Is there end to end encryption?',
            'relevant_doc_ids': ['doc_1_chunk_72', 'doc_3_chunk_10', 'doc_3_chunk_11'],
            'ground_truth': 'Yes, Enterprise customers can enable end-to-end encryption for sensitive projects.'
        }
    ]
    
    # Create reporter and evaluate
    reporter = RagEvaluator()
    reporter.initialize_rag_system()
    metrics = reporter.evaluate_rag_system(test_queries, n_results=3)
    
    # Print and export results
    reporter.print_console_report()
    
    print("\n" + "="*70)
    print("Performance evaluation complete!")
    print("="*70)
