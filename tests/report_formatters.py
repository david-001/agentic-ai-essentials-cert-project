"""
Report formatting utilities for performance metrics.
Extracted from performance_reporter.py to reduce file size.
"""

from typing import Dict


def format_markdown_report(metrics: Dict[str, float], timestamp: str, 
                          thresholds: Dict, summary: str, recommendations: str) -> str:
    """Generate markdown formatted report."""
    lines = []
    lines.append("# RAG System Performance Report")
    lines.append(f"\n**Generated:** {timestamp}\n")
    lines.append("---\n")
    
    # Summary
    lines.append("## Executive Summary\n")
    lines.append(summary)
    lines.append("\n---\n")
    
    # Retrieval metrics
    if 'precision_at_3' in metrics:
        lines.append("## Retrieval Metrics\n")
        lines.append("| Metric | Value | Status |")
        lines.append("|--------|-------|--------|")
        
        retrieval_metrics = [
            ('Precision@3', 'precision_at_3'),
            ('Recall@3', 'recall_at_3'),
            ('MRR', 'mrr'),
            ('NDCG@5', 'ndcg_at_5')
        ]
        
        for name, key in retrieval_metrics:
            if key in metrics:
                value = metrics[key]
                status = get_status(key, value, thresholds)
                emoji = get_status_emoji(status)
                lines.append(f"| {name} | {value:.4f} | {emoji} {status} |")
        
        if 'avg_latency_ms' in metrics:
            value = metrics['avg_latency_ms']
            status = get_status('avg_latency_ms', value, thresholds)
            emoji = get_status_emoji(status)
            lines.append(f"| Avg Latency | {value:.2f}ms | {emoji} {status} |")
        
        lines.append("")
    
    # Generation metrics
    if 'faithfulness' in metrics:
        lines.append("## Generation Metrics\n")
        lines.append("| Metric | Value | Status |")
        lines.append("|--------|-------|--------|")
        
        generation_metrics = [
            ('Faithfulness', 'faithfulness'),
            ('Answer Relevance', 'answer_relevance'),
            ('Hallucination Rate', 'hallucination_rate'),
            ('Ground Truth Overlap', 'ground_truth_overlap')
        ]
        
        for name, key in generation_metrics:
            if key in metrics:
                value = metrics[key]
                status = get_status(key, value, thresholds)
                emoji = get_status_emoji(status)
                
                if key == 'hallucination_rate':
                    lines.append(f"| {name} | {value:.2%} | {emoji} {status} |")
                else:
                    lines.append(f"| {name} | {value:.4f} | {emoji} {status} |")
        
        lines.append("")
    
    # Recommendations
    lines.append("## Recommendations\n")
    lines.append(recommendations)
    lines.append("")
    
    # Threshold reference
    lines.append("---")
    lines.append("\n### Performance Thresholds\n")
    lines.append("- 🟢 **Excellent**: Exceeds high-quality threshold")
    lines.append("- 🟡 **Good**: Meets acceptable threshold")
    lines.append("- 🔴 **Needs Improvement**: Below acceptable threshold")
    
    return "\n".join(lines)


def get_status(metric: str, value: float, thresholds: Dict[str, Dict[str, float]]) -> str:
    """Determine status (Excellent/Good/Needs Improvement) for a metric."""
    if metric not in thresholds:
        return "N/A"
    
    thresh = thresholds[metric]
    
    # For metrics where lower is better
    if metric in ['hallucination_rate', 'avg_latency_ms']:
        if value <= thresh['excellent']:
            return "Excellent"
        elif value <= thresh['good']:
            return "Good"
        else:
            return "Needs Improvement"
    else:
        # For metrics where higher is better
        if value >= thresh['excellent']:
            return "Excellent"
        elif value >= thresh['good']:
            return "Good"
        else:
            return "Needs Improvement"


def get_status_emoji(status: str) -> str:
    """Get emoji for status."""
    return {
        'Excellent': '🟢',
        'Good': '🟡',
        'Needs Improvement': '🔴',
        'N/A': '⚪'
    }.get(status, '')
