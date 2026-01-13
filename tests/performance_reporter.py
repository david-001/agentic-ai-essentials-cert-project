"""
Performance Report Generator for RAG System.
Generates detailed performance reports in JSON, CSV, and Markdown formats.
"""

import json
import csv
import os
import sys
import numpy as np
import random
from datetime import datetime
from typing import Dict, List, Any

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from report_formatters import format_markdown_report, get_status, get_status_emoji


class PerformanceReporter:
    """Generate and export performance metrics reports."""
    
    def __init__(self):
        self.metrics = {}
        self.timestamp = datetime.now().isoformat()
    
    def calculate_all_metrics(
        self,
        retrieval_results: List[Dict[str, Any]],
        generation_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics.
        
        Args:
            retrieval_results: List of retrieval results with keys:
                - retrieved_ids: List of retrieved document IDs
                - relevant_ids: List of relevant document IDs
                - query: The query string
                - latency: Retrieval latency in seconds
            
            generation_results: List of generation results with keys:
                - answer: Generated answer
                - context: Retrieved context
                - ground_truth: Expected answer
                - query: The query string
        
        Returns:
            Dictionary of all calculated metrics
        """
        metrics = {}
        
        # Retrieval metrics
        if retrieval_results:
            metrics.update(self._calculate_retrieval_metrics(retrieval_results))
        
        # Generation metrics
        if generation_results:
            metrics.update(self._calculate_generation_metrics(generation_results))
        
        self.metrics = metrics
        return metrics
    
    def _calculate_retrieval_metrics(
        self, 
        results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate retrieval-specific metrics."""
        k = 3
        metrics = {
            'precision_at_3': [],
            'recall_at_3': [],
            'mrr': [],
            'ndcg_at_5': [],
            'avg_latency_ms': []
        }
        
        for result in results:
            retrieved = result['retrieved_ids']
            relevant = result['relevant_ids']
            
            # Precision@k
            top_k = retrieved[:k]
            precision = len(set(top_k) & set(relevant)) / k if k > 0 else 0
            metrics['precision_at_3'].append(precision)
            
            # Recall@k
            recall = (len(set(top_k) & set(relevant)) / len(relevant) 
                     if relevant else 0)
            metrics['recall_at_3'].append(recall)
            
            # MRR
            rr = 0
            for rank, doc_id in enumerate(retrieved, start=1):
                if doc_id in relevant:
                    rr = 1.0 / rank
                    break
            metrics['mrr'].append(rr)
            
            # NDCG@5
            relevances = [1.0 if doc_id in relevant else 0.0 
                         for doc_id in retrieved[:5]]
            dcg = sum(rel / np.log2(i + 1) 
                     for i, rel in enumerate(relevances, start=1))
            ideal_rel = sorted(relevances, reverse=True)
            idcg = sum(rel / np.log2(i + 1) 
                      for i, rel in enumerate(ideal_rel, start=1))
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics['ndcg_at_5'].append(ndcg)
            
            # Latency
            if 'latency' in result:
                metrics['avg_latency_ms'].append(result['latency'] * 1000)
        
        # Calculate averages
        return {
            'precision_at_3': np.mean(metrics['precision_at_3']),
            'recall_at_3': np.mean(metrics['recall_at_3']),
            'mrr': np.mean(metrics['mrr']),
            'ndcg_at_5': np.mean(metrics['ndcg_at_5']),
            'avg_latency_ms': np.mean(metrics['avg_latency_ms']) 
                if metrics['avg_latency_ms'] else 0
        }
    
    def _calculate_generation_metrics(
        self, 
        results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate generation-specific metrics."""
        metrics = {
            'faithfulness': [],
            'answer_relevance': [],
            'hallucination_count': 0,
            'token_overlap_with_gt': []
        }
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 
                      'at', 'to', 'for', 'is', 'are', 'was', 'were'}
        
        for result in results:
            answer = result['answer']
            context = result['context']
            query = result['query']
            
            # Faithfulness
            answer_tokens = set(answer.lower().split()) - stop_words
            context_tokens = set(context.lower().split()) - stop_words
            
            faithfulness = (len(answer_tokens & context_tokens) / len(answer_tokens) 
                           if answer_tokens else 0)
            metrics['faithfulness'].append(faithfulness)
            
            # Hallucination detection
            if faithfulness < 0.5:
                metrics['hallucination_count'] += 1
            
            # Answer relevance
            query_tokens = set(query.lower().split()) - stop_words
            all_tokens = answer_tokens | query_tokens
            relevance = (len(answer_tokens & query_tokens) / len(all_tokens) 
                        if all_tokens else 0)
            metrics['answer_relevance'].append(relevance)
            
            # Ground truth overlap
            if 'ground_truth' in result:
                gt_tokens = set(result['ground_truth'].lower().split()) - stop_words
                all_gt_tokens = answer_tokens | gt_tokens
                overlap = (len(answer_tokens & gt_tokens) / len(all_gt_tokens) 
                          if all_gt_tokens else 0)
                metrics['token_overlap_with_gt'].append(overlap)
        
        return {
            'faithfulness': np.mean(metrics['faithfulness']),
            'answer_relevance': np.mean(metrics['answer_relevance']),
            'hallucination_rate': metrics['hallucination_count'] / len(results),
            'ground_truth_overlap': (np.mean(metrics['token_overlap_with_gt']) 
                                    if metrics['token_overlap_with_gt'] else 0)
        }
    
    def generate_report_json(self, filepath: str = None) -> str:
        """
        Generate JSON report.
        
        Args:
            filepath: Optional path to save JSON file
        
        Returns:
            JSON string
        """
        report = {
            'timestamp': self.timestamp,
            'metrics': self.metrics,
            'summary': self._generate_summary()
        }
        
        json_str = json.dumps(report, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            print(f"JSON report saved to: {filepath}")
        
        return json_str
    
    def generate_report_csv(self, filepath: str):
        """Generate CSV report."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value', 'Status'])
            
            thresholds = self._get_thresholds()
            
            for metric, value in self.metrics.items():
                status = get_status(metric, value, thresholds)
                writer.writerow([metric, f"{value:.4f}", status])
        
        print(f"CSV report saved to: {filepath}")
    
    def generate_report_markdown(self, filepath: str = None) -> str:
        """
        Generate markdown report.
        
        Args:
            filepath: Optional path to save markdown file
        
        Returns:
            Markdown string
        """
        thresholds = self._get_thresholds()
        summary = self._generate_summary()
        recommendations = self._generate_recommendations()
        
        markdown = format_markdown_report(
            self.metrics, 
            self.timestamp, 
            thresholds, 
            summary, 
            recommendations
        )
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(markdown)
            print(f"Markdown report saved to: {filepath}")
        
        return markdown
    
    def _generate_summary(self) -> str:
        """Generate text summary of metrics."""
        if not self.metrics:
            return "No metrics calculated yet."
        
        lines = []
        
        # Overall assessment
        grade = self._calculate_grade()
        lines.append(f"**Overall Performance:** {grade}")
        lines.append("")
        
        # Key metrics
        if 'precision_at_3' in self.metrics:
            lines.append(f"- Retrieval precision: {self.metrics['precision_at_3']:.2%}")
        if 'faithfulness' in self.metrics:
            lines.append(f"- Answer faithfulness: {self.metrics['faithfulness']:.2%}")
        if 'hallucination_rate' in self.metrics:
            lines.append(f"- Hallucination rate: {self.metrics['hallucination_rate']:.2%}")
        
        return "\n".join(lines)
    
    def _get_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get threshold values for each metric."""
        return {
            'precision_at_3': {'excellent': 0.8, 'good': 0.6},
            'recall_at_3': {'excellent': 0.8, 'good': 0.6},
            'mrr': {'excellent': 0.8, 'good': 0.7},
            'ndcg_at_5': {'excellent': 0.85, 'good': 0.7},
            'faithfulness': {'excellent': 0.8, 'good': 0.7},
            'answer_relevance': {'excellent': 0.7, 'good': 0.5},
            'hallucination_rate': {'excellent': 0.05, 'good': 0.10},
            'avg_latency_ms': {'excellent': 200, 'good': 500}
        }
    
    def _calculate_grade(self) -> str:
        """Calculate overall grade."""
        if not self.metrics:
            return "No data"
        
        thresholds = self._get_thresholds()
        scores = []
        
        for metric, value in self.metrics.items():
            status = get_status(metric, value, thresholds)
            if status == "Excellent":
                scores.append(3)
            elif status == "Good":
                scores.append(2)
            elif status == "Needs Improvement":
                scores.append(1)
        
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
    
    def _generate_recommendations(self) -> str:
        """Generate improvement recommendations."""
        recommendations = []
        thresholds = self._get_thresholds()
        
        for metric, value in self.metrics.items():
            status = get_status(metric, value, thresholds)
            
            if status == "Needs Improvement":
                if metric == 'precision_at_3':
                    recommendations.append(
                        "- **Low Precision**: Consider using better embedding models "
                        "or fine-tuning on your domain."
                    )
                elif metric == 'recall_at_3':
                    recommendations.append(
                        "- **Low Recall**: Increase number of retrieved documents (k) "
                        "or improve chunking strategy."
                    )
                elif metric == 'faithfulness':
                    recommendations.append(
                        "- **Low Faithfulness**: Improve prompt to emphasize using "
                        "only provided context. Add citation requirements."
                    )
                elif metric == 'hallucination_rate':
                    recommendations.append(
                        "- **High Hallucination Rate**: Use stricter prompts and "
                        "implement fact-checking mechanisms."
                    )
                elif metric == 'avg_latency_ms':
                    recommendations.append(
                        "- **High Latency**: Optimize vector search, reduce chunk "
                        "size, or implement caching."
                    )
        
        if not recommendations:
            recommendations.append("- System performing well! Continue monitoring.")
        
        return "\n".join(recommendations)
    
    def print_console_report(self):
        """Print formatted report to console."""
        print("\n" + "="*70)
        print("RAG SYSTEM PERFORMANCE REPORT")
        print("="*70)
        print(f"\nGenerated: {self.timestamp}")
        print("\n" + self._generate_summary())
        print("\n" + "="*70)
        
        if 'precision_at_3' in self.metrics:
            print("\nRETRIEVAL METRICS:")
            print(f"  Precision@3:  {self.metrics['precision_at_3']:.4f}")
            print(f"  Recall@3:     {self.metrics['recall_at_3']:.4f}")
            print(f"  MRR:          {self.metrics.get('mrr', 0):.4f}")
            print(f"  NDCG@5:       {self.metrics.get('ndcg_at_5', 0):.4f}")
            if 'avg_latency_ms' in self.metrics:
                print(f"  Avg Latency:  {self.metrics['avg_latency_ms']:.2f}ms")
        
        if 'faithfulness' in self.metrics:
            print("\nGENERATION METRICS:")
            print(f"  Faithfulness:     {self.metrics['faithfulness']:.4f}")
            print(f"  Answer Relevance: {self.metrics['answer_relevance']:.4f}")
            print(f"  Hallucination:    {self.metrics['hallucination_rate']:.2%}")
        
        print("\n" + "="*70)
        print(f"\nOVERALL GRADE: {self._calculate_grade()}")
        print("\n" + "="*70)


# Example usage
if __name__ == "__main__":
    # Example data
    retrieval_results = [
        {
            'retrieved_ids': ['doc_0_chunk_0', 'doc_0_chunk_1', 'doc_1_chunk_0'],
            'relevant_ids': ['doc_0_chunk_0', 'doc_0_chunk_1'],
            'query': 'Test query 1',
            'latency': 0.15
        },
        {
            'retrieved_ids': ['doc_1_chunk_0', 'doc_1_chunk_1', 'doc_0_chunk_0'],
            'relevant_ids': ['doc_1_chunk_0', 'doc_1_chunk_1'],
            'query': 'Test query 2',
            'latency': 0.12
        }
    ]
    
    generation_results = [
        {
            'answer': 'This is a test answer about vacation days.',
            'context': 'Company policy states vacation days are based on tenure.',
            'query': 'How many vacation days?',
            'ground_truth': 'Vacation days depend on tenure'
        },
        {
            'answer': 'API authentication uses OAuth 2.0.',
            'context': 'The API supports OAuth 2.0 and API key authentication.',
            'query': 'What authentication methods?',
            'ground_truth': 'OAuth 2.0 and API keys'
        }
    ]
    
    # Generate report
    reporter = PerformanceReporter()
    metrics = reporter.calculate_all_metrics(retrieval_results, generation_results)
    
    # Print to console
    reporter.print_console_report()
    
    # Export to files
    reporter.generate_report_json('performance_report.json')
    reporter.generate_report_csv('performance_report.csv')
    reporter.generate_report_markdown('performance_report.md')
