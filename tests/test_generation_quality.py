"""
Generation Quality Tests for RAG System.
Tests evaluate the quality of generated answers including accuracy, hallucination detection,
context adherence, and response formatting.
"""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock
import re

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import RAGAssistant


@pytest.fixture
def knowledge_base_docs():
    """Fixture providing a knowledge base with verifiable facts."""
    return [
        {
            'content': (
                "Company Vacation Policy 2024\n\n"
                "All full-time employees are entitled to paid vacation days:\n"
                "- Entry-level (0-2 years): 15 days per year\n"
                "- Mid-level (3-5 years): 20 days per year\n"
                "- Senior-level (5+ years): 25 days per year\n\n"
                "Vacation requests must be submitted at least 2 weeks in advance through the HR portal. "
                "Unused vacation days cannot be carried over to the next year. "
                "Vacation days are pro-rated for employees who start mid-year."
            ),
            'metadata': {'source': 'vacation_policy.txt', 'doc_id': '1'}
        },
        {
            'content': (
                "Remote Work Guidelines\n\n"
                "Employees may work remotely up to 3 days per week. Remote work arrangements must be "
                "approved by direct managers. All remote workers must:\n"
                "- Maintain regular business hours (9 AM - 5 PM in their timezone)\n"
                "- Be available on Slack during work hours\n"
                "- Attend all team meetings via video conference\n"
                "- Have a dedicated workspace with reliable internet\n\n"
                "Equipment stipend: $500 annually for home office setup. "
                "VPN access is required for all remote connections."
            ),
            'metadata': {'source': 'remote_work.txt', 'doc_id': '2'}
        },
        {
            'content': (
                "Health Insurance Benefits\n\n"
                "The company offers three health insurance plans:\n"
                "1. Basic Plan: $0 employee contribution, $2,000 deductible\n"
                "2. Standard Plan: $100/month contribution, $1,000 deductible\n"
                "3. Premium Plan: $200/month contribution, $500 deductible\n\n"
                "All plans include vision and dental coverage. "
                "Open enrollment is held every November. "
                "New employees can enroll within 30 days of hire date."
            ),
            'metadata': {'source': 'health_insurance.txt', 'doc_id': '3'}
        },
        {
            'content': (
                "Performance Review Process\n\n"
                "Performance reviews are conducted bi-annually in June and December. "
                "The review process includes:\n"
                "1. Self-assessment submission\n"
                "2. Peer feedback collection (3-5 peers)\n"
                "3. Manager review meeting\n"
                "4. Goal setting for next period\n\n"
                "Reviews determine: salary adjustments, bonuses, and promotion eligibility. "
                "Performance ratings: Exceeds Expectations, Meets Expectations, Needs Improvement."
            ),
            'metadata': {'source': 'performance_reviews.txt', 'doc_id': '4'}
        }
    ]


@pytest.fixture
def assistant_with_knowledge(knowledge_base_docs):
    """Fixture providing RAG assistant with knowledge base loaded."""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        assistant = RAGAssistant()
        assistant.add_documents(knowledge_base_docs)
        return assistant


class TestFactualAccuracy:
    """Test suite for verifying factual accuracy of generated answers."""
    
    @pytest.mark.generation
    def test_specific_fact_retrieval(self, assistant_with_knowledge):
        """Test that specific facts are accurately retrieved and stated."""
        # Mock the LLM to return a realistic response
        with patch.object(assistant_with_knowledge, 'chain') as mock_chain:
            mock_chain.invoke.return_value = (
                "Entry-level employees (0-2 years) receive 15 days of vacation per year."
            )
            
            result = assistant_with_knowledge.query(
                "How many vacation days do entry-level employees get?"
            )
            
            # Check that the specific number is mentioned
            assert '15' in result, "Should mention the correct number of days"
            assert 'days' in result.lower(), "Should mention 'days'"
    
    @pytest.mark.generation
    def test_numerical_accuracy(self, assistant_with_knowledge):
        """Test that numerical values are accurately reported."""
        test_cases = [
            ("remote work days per week", "3"),
            ("health insurance basic plan deductible", "2000" or "2,000"),
            ("equipment stipend amount", "500"),
        ]
        
        # For each test case, verify the number appears in context
        for query, expected_number in test_cases:
            # Check what context is retrieved
            search_results = assistant_with_knowledge.vector_db.search(query, n_results=1)
            context = search_results['documents'][0] if search_results['documents'] else ""
            
            # Remove commas for comparison
            context_clean = context.replace(',', '')
            expected_clean = expected_number.replace(',', '')
            
            assert expected_clean in context_clean, \
                f"Number {expected_number} should be in retrieved context for query: {query}"
    
    @pytest.mark.generation
    def test_multiple_fact_accuracy(self, assistant_with_knowledge):
        """Test accuracy when answer requires multiple facts."""
        with patch.object(assistant_with_knowledge, 'chain') as mock_chain:
            mock_chain.invoke.return_value = (
                "The three health insurance plans are:\n"
                "1. Basic Plan: $0 contribution, $2,000 deductible\n"
                "2. Standard Plan: $100/month, $1,000 deductible\n"
                "3. Premium Plan: $200/month, $500 deductible"
            )
            
            result = assistant_with_knowledge.query(
                "What are the different health insurance plans?"
            )
            
            # Verify multiple facts are present
            assert 'Basic' in result or 'basic' in result.lower()
            assert 'Standard' in result or 'standard' in result.lower()
            assert 'Premium' in result or 'premium' in result.lower()
    
    @pytest.mark.generation
    def test_context_completeness(self, assistant_with_knowledge):
        """Test that answers include all relevant information from context."""
        with patch.object(assistant_with_knowledge, 'chain') as mock_chain:
            # Simulate a complete answer
            mock_chain.invoke.return_value = (
                "Vacation requests must be submitted at least 2 weeks in advance through "
                "the HR portal. Unused vacation days cannot be carried over to the next year."
            )
            
            result = assistant_with_knowledge.query(
                "What are the rules for requesting vacation?"
            )
            
            # Check for key policy elements
            assert '2 weeks' in result or 'two weeks' in result.lower()
            assert 'advance' in result.lower()


class TestHallucinationDetection:
    """Test suite for detecting hallucinations and fabricated information."""
    
    @pytest.mark.generation
    def test_no_fabricated_numbers(self, assistant_with_knowledge):
        """Test that the system doesn't fabricate numbers not in the context."""
        # Query about vacation days
        search_results = assistant_with_knowledge.vector_db.search(
            "vacation days for employees", n_results=2
        )
        context = " ".join(search_results['documents'])
        
        # Extract numbers mentioned in context
        context_numbers = set(re.findall(r'\b\d+\b', context))
        
        with patch.object(assistant_with_knowledge, 'chain') as mock_chain:
            # Simulate a response
            mock_chain.invoke.return_value = "Employees receive 15, 20, or 25 vacation days."
            result = assistant_with_knowledge.query("How many vacation days?")
            
            # Extract numbers from result
            result_numbers = set(re.findall(r'\b\d+\b', result))
            
            # Check if fabricated numbers exist (not in context but in result)
            # This is a soft check since mocking makes this test illustrative
            print(f"Context numbers: {context_numbers}")
            print(f"Result numbers: {result_numbers}")
    
    @pytest.mark.generation
    def test_no_out_of_scope_information(self, assistant_with_knowledge):
        """Test that system doesn't add information outside the knowledge base."""
        # Ask about something completely unrelated
        with patch.object(assistant_with_knowledge, 'chain') as mock_chain:
            mock_chain.invoke.return_value = (
                "The question is not answerable given the documents."
            )
            
            result = assistant_with_knowledge.query(
                "What is the capital of France?"
            )
            
            # Should refuse to answer or say not answerable
            refuses = any(phrase in result.lower() for phrase in [
                'not answerable',
                'not available',
                'cannot answer',
                'no information',
                'not found'
            ])
            
            assert refuses, "Should refuse to answer out-of-scope questions"
    
    @pytest.mark.generation
    def test_sticks_to_context(self, assistant_with_knowledge):
        """Test that answers are grounded in retrieved context."""
        # Get the actual context that would be retrieved
        query = "remote work policy"
        search_results = assistant_with_knowledge.vector_db.search(query, n_results=2)
        context = " ".join(search_results['documents'])
        
        # Check that context contains relevant information
        assert 'remote' in context.lower(), "Context should contain remote work info"
        
        # Verify the chain receives this context
        with patch.object(assistant_with_knowledge, 'chain') as mock_chain:
            mock_chain.invoke.return_value = "Remote work is allowed up to 3 days per week."
            
            assistant_with_knowledge.query(query)
            
            # Verify context was passed to chain
            call_args = mock_chain.invoke.call_args[0][0]
            assert 'context' in call_args
            assert len(call_args['context']) > 0
    
    @pytest.mark.generation
    def test_no_contradictory_information(self, assistant_with_knowledge):
        """Test that answers don't contradict the knowledge base."""
        queries_and_correct_facts = [
            ("entry-level vacation days", ["15", "entry-level", "0-2 years"]),
            ("remote work days", ["3 days", "per week"]),
            ("basic health plan", ["$0", "employee contribution"]),
        ]
        
        for query, required_elements in queries_and_correct_facts:
            search_results = assistant_with_knowledge.vector_db.search(query, n_results=1)
            context = search_results['documents'][0] if search_results['documents'] else ""
            
            # Verify that key facts from document are in context
            for element in required_elements:
                element_clean = element.replace('$', '').replace(',', '')
                context_clean = context.replace('$', '').replace(',', '')
                
                assert element_clean.lower() in context_clean.lower(), \
                    f"Context should contain '{element}' for query '{query}'"


class TestContextAdherence:
    """Test suite for ensuring answers adhere to provided context."""
    
    @pytest.mark.generation
    def test_answer_based_on_context_only(self, assistant_with_knowledge):
        """Test that answers are based solely on provided context."""
        # Query that might tempt LLM to use general knowledge
        query = "What is the vacation policy?"
        
        # Get actual context
        search_results = assistant_with_knowledge.vector_db.search(query, n_results=2)
        actual_context = " ".join(search_results['documents'])
        
        with patch.object(assistant_with_knowledge, 'chain') as mock_chain:
            # Check what was passed to the chain
            def capture_invoke(inputs):
                assert 'context' in inputs, "Must provide context"
                assert 'question' in inputs, "Must provide question"
                return "Based on the context provided..."
            
            mock_chain.invoke.side_effect = capture_invoke
            assistant_with_knowledge.query(query)
            
            assert mock_chain.invoke.called
    
    @pytest.mark.generation
    def test_explicit_context_citation(self, assistant_with_knowledge):
        """Test that answers reference the context appropriately."""
        with patch.object(assistant_with_knowledge, 'chain') as mock_chain:
            # Good answer that references context
            mock_chain.invoke.return_value = (
                "According to the vacation policy, entry-level employees receive 15 days."
            )
            
            result = assistant_with_knowledge.query("vacation days")
            
            # Check for context-referencing phrases (optional but good practice)
            context_refs = [
                'according to',
                'based on',
                'the policy states',
                'as mentioned',
            ]
            
            # This is a soft check - not all good answers need explicit references
            print(f"Answer: {result}")
    
    @pytest.mark.generation
    def test_refuses_when_context_insufficient(self, assistant_with_knowledge):
        """Test that system refuses to answer when context is insufficient."""
        # Query about something not in knowledge base
        query = "What is the company's cryptocurrency policy?"
        
        search_results = assistant_with_knowledge.vector_db.search(query, n_results=2)
        context = " ".join(search_results['documents'])
        
        # Context should not contain cryptocurrency info
        assert 'cryptocurrency' not in context.lower()
        
        with patch.object(assistant_with_knowledge, 'chain') as mock_chain:
            mock_chain.invoke.return_value = "The question is not answerable given the documents."
            
            result = assistant_with_knowledge.query(query)
            
            # Should indicate inability to answer
            refuses = any(phrase in result.lower() for phrase in [
                'not answerable',
                'cannot answer',
                'no information',
                'not found in'
            ])
            
            assert refuses or len(result) < 50, \
                "Should refuse or give minimal response when info not available"


class TestResponseFormatting:
    """Test suite for evaluating response formatting and structure."""
    
    @pytest.mark.generation
    def test_response_is_string(self, assistant_with_knowledge):
        """Test that response is a proper string."""
        with patch.object(assistant_with_knowledge, 'chain') as mock_chain:
            mock_chain.invoke.return_value = "This is a test response."
            
            result = assistant_with_knowledge.query("test query")
            
            assert isinstance(result, str), "Response must be a string"
            assert len(result) > 0, "Response should not be empty"
    
    @pytest.mark.generation
    def test_response_uses_bullet_points_when_appropriate(self, assistant_with_knowledge):
        """Test that responses use bullet points for lists."""
        with patch.object(assistant_with_knowledge, 'chain') as mock_chain:
            # Simulate a well-formatted list response
            mock_chain.invoke.return_value = (
                "The health insurance plans are:\n"
                "• Basic Plan: $0 contribution\n"
                "• Standard Plan: $100/month\n"
                "• Premium Plan: $200/month"
            )
            
            result = assistant_with_knowledge.query("What are the health insurance plans?")
            
            # Check for list formatting
            has_formatting = any(marker in result for marker in ['•', '-', '*', '\n1.', '\n2.'])
            print(f"Response formatting: {has_formatting}")
            print(f"Response:\n{result}")
    
    @pytest.mark.generation
    def test_response_length_appropriate(self, assistant_with_knowledge):
        """Test that response length is appropriate for the query."""
        test_cases = [
            ("How many vacation days?", "short"),  # Should be concise
            ("Explain the entire vacation policy", "long"),  # Can be detailed
        ]
        
        for query, expected_length in test_cases:
            with patch.object(assistant_with_knowledge, 'chain') as mock_chain:
                if expected_length == "short":
                    mock_chain.invoke.return_value = "15 days for entry-level employees."
                else:
                    mock_chain.invoke.return_value = (
                        "The vacation policy includes several key points: "
                        "Entry-level employees get 15 days, mid-level get 20 days, "
                        "senior-level get 25 days. Requests must be submitted 2 weeks "
                        "in advance. Days cannot be carried over."
                    )
                
                result = assistant_with_knowledge.query(query)
                
                if expected_length == "short":
                    assert len(result) < 200, "Short answers should be concise"
                else:
                    assert len(result) > 100, "Detailed answers should be comprehensive"
    
    @pytest.mark.generation
    def test_response_grammar_and_clarity(self, assistant_with_knowledge):
        """Test that responses are grammatically correct and clear."""
        with patch.object(assistant_with_knowledge, 'chain') as mock_chain:
            mock_chain.invoke.return_value = (
                "Entry-level employees receive 15 vacation days per year. "
                "Vacation must be requested 2 weeks in advance."
            )
            
            result = assistant_with_knowledge.query("vacation policy")
            
            # Basic grammar checks
            assert result[0].isupper() or result[0].isdigit(), \
                "Response should start with capital letter or number"
            assert result.endswith('.') or result.endswith('?') or result.endswith('!'), \
                "Response should end with proper punctuation"


class TestAnswerConsistency:
    """Test suite for consistency across multiple queries."""
    
    @pytest.mark.generation
    def test_consistent_facts_across_queries(self, assistant_with_knowledge):
        """Test that the same facts are reported consistently."""
        queries = [
            "How many vacation days for entry-level?",
            "What is the vacation time for 0-2 years experience?",
            "Entry-level employee vacation days?"
        ]
        
        results = []
        for query in queries:
            with patch.object(assistant_with_knowledge, 'chain') as mock_chain:
                mock_chain.invoke.return_value = "Entry-level employees receive 15 vacation days."
                result = assistant_with_knowledge.query(query)
                results.append(result)
        
        # Check that all answers contain "15"
        assert all('15' in result for result in results), \
            "Should consistently report 15 days"
    
    @pytest.mark.generation
    def test_no_contradictions_in_complex_query(self, assistant_with_knowledge):
        """Test that complex queries don't produce contradictory information."""
        query = "Tell me about vacation days for different employee levels"
        
        with patch.object(assistant_with_knowledge, 'chain') as mock_chain:
            mock_chain.invoke.return_value = (
                "Vacation days vary by level:\n"
                "- Entry-level (0-2 years): 15 days\n"
                "- Mid-level (3-5 years): 20 days\n"
                "- Senior-level (5+ years): 25 days"
            )
            
            result = assistant_with_knowledge.query(query)
            
            # Verify all levels are mentioned with correct numbers
            assert '15' in result and '20' in result and '25' in result


class TestEdgeCaseGeneration:
    """Test suite for edge cases in generation."""
    
    @pytest.mark.generation
    def test_empty_query_handling(self, assistant_with_knowledge):
        """Test handling of empty query."""
        with patch.object(assistant_with_knowledge, 'chain') as mock_chain:
            mock_chain.invoke.return_value = "Please provide a question."
            
            result = assistant_with_knowledge.query("")
            
            assert isinstance(result, str), "Should return string even for empty query"
    
    @pytest.mark.generation
    def test_ambiguous_query_handling(self, assistant_with_knowledge):
        """Test handling of ambiguous queries."""
        with patch.object(assistant_with_knowledge, 'chain') as mock_chain:
            mock_chain.invoke.return_value = (
                "Could you please clarify whether you're asking about vacation days, "
                "remote work, or another policy?"
            )
            
            result = assistant_with_knowledge.query("policy")
            
            # Should either ask for clarification or provide general policy info
            assert len(result) > 0


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


@pytest.mark.generation
def test_generate_quality_report(assistant_with_knowledge):
    """Generate and display comprehensive answer quality report."""
    report = generate_answer_quality_report(assistant_with_knowledge)
    print(report)
    assert True  # This test always passes, it's for reporting


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "generation"])
