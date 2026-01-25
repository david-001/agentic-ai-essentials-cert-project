"""
Helper utilities for rag_evaluator.py
"""

from typing import Dict

def get_status(metric: str, value: float, thresholds: Dict[str, Dict[str, float]]) -> str:
    """Determine status (Excellent/Good/Needs Improvement) for a metric."""
    if metric not in thresholds:
        return "N/A"
    
    thresh = thresholds[metric]
    
    # For metrics where lower is better
    if metric == 'avg_latency_ms':
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