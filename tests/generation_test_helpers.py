"""
Helper utilities for generation quality tests.
Extracted from test_generation_quality.py to reduce file size.
"""

from typing import List


def generate_answer_quality_report(assistant_with_knowledge) -> str:
    """Generate a comprehensive answer quality report."""
    report = "\n" + "="*60 + "\n"
    report += "ANSWER QUALITY REPORT\n"
    report += "="*60 + "\n\n"
    
    test_queries = [
        "How many vacation days do entry-level employees get?",
        "What is the remote work policy?",
        "What are the health insurance options?",
        "When are performance reviews conducted?"
    ]
    
    report += "Test Queries:\n"
    for i, query in enumerate(test_queries, 1):
        report += f"{i}. {query}\n"
    
    report += "\nAnswers Generated:\n"
    report += "(Note: Answers depend on LLM being mocked or real)\n\n"
    
    # Check context retrieval quality
    report += "Context Retrieval Check:\n"
    for query in test_queries:
        results = assistant_with_knowledge.vector_db.search(query, n_results=1)
        has_context = len(results['documents']) > 0 and len(results['documents'][0]) > 0
        report += f"  {query[:40]+'...' if len(query) > 40 else query}: "
        report += f"{'✓ Context found' if has_context else '✗ No context'}\n"
    
    report += "\n" + "="*60 + "\n"
    
    return report


def check_refusal_phrases(text: str) -> bool:
    """Check if text contains refusal phrases."""
    refusal_phrases = [
        'not answerable',
        'not available',
        'cannot answer',
        'no information',
        'not found',
        'not found in'
    ]
    return any(phrase in text.lower() for phrase in refusal_phrases)


def check_context_reference_phrases(text: str) -> List[str]:
    """Check for context-referencing phrases in text."""
    context_refs = [
        'according to',
        'based on',
        'the policy states',
        'as mentioned',
    ]
    return [phrase for phrase in context_refs if phrase in text.lower()]
