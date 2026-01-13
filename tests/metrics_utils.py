"""
Utility functions for calculating performance metrics.
Extracted from test_performance_metrics.py to reduce file size.
"""

import numpy as np
from typing import List, Tuple


def calculate_precision_at_k(
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


def calculate_reciprocal_rank(
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


def calculate_mrr(queries_results: List[Tuple[List[str], List[str]]]) -> float:
    """
    Calculate Mean Reciprocal Rank across multiple queries.
    queries_results: List of (retrieved_ids, relevant_ids) tuples
    """
    rr_scores = [
        calculate_reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in queries_results
    ]
    return np.mean(rr_scores) if rr_scores else 0.0


def calculate_dcg(relevances: List[float], k: int = None) -> float:
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
    dcg = calculate_dcg(relevances, k)
    
    # Calculate ideal DCG (perfect ranking)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)
    
    # Calculate NDCG
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_token_overlap(text1: str, text2: str) -> float:
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


def calculate_faithfulness(answer: str, context: str) -> float:
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


def calculate_answer_relevance(answer: str, question: str) -> float:
    """
    Calculate answer relevance score.
    Measures if answer addresses the question.
    """
    return calculate_token_overlap(answer, question)
