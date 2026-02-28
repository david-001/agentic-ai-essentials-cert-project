"""
synthesize_test_queries.py

Uses deepeval.synthesizer to automatically generate test queries
with 'query', 'relevant_doc_ids', and 'ground_truth' fields
from the project's markdown documents — matching the exact format
expected by RagEvaluator.evaluate_rag_system().

Usage:
    python tests/synthesize_test_queries.py

Output:
    - Prints generated test_queries to console
    - Saves to tests/synthesized_test_queries.json
"""

import os
import sys
import json

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from deepeval.synthesizer import Synthesizer
from deepeval.models import GPTModel, GeminiModel

# ── Config ────────────────────────────────────────────────────────────────────

# Match the chunking parameters in VectorDB.chunk_text()
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50

# Number of Q&A pairs generated per chunk (increase for more test cases)
MAX_GOLDENS_PER_CONTEXT = 1

# Source documents — order determines doc index (doc_0, doc_1, ...)
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

DOC_FILES = [
    'api_documentation.md',     # doc_0
    'customer_faq.md',          # doc_1
    'company_policies.md',      # doc_2
    'security_compliance.md',   # doc_3
    'product_documentation.md', # doc_4
]

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'synthesized_test_queries.json')


# ── Model initialisation ──────────────────────────────────────────────────────

def _init_model():
    """Initialise the same LLM used by RagEvaluator."""
    if os.getenv("OPENAI_API_KEY"):
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        print(f"✓ Using OpenAI: {model_name}")
        return GPTModel(model=model_name, api_key=os.getenv("OPENAI_API_KEY"))

    elif os.getenv("GROQ_API_KEY"):
        model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        print(f"✓ Using Groq: {model_name}")
        return GPTModel(
            model=model_name,
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )

    elif os.getenv("GOOGLE_API_KEY"):
        model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
        print(f"✓ Using Gemini: {model_name}")
        return GeminiModel(model=model_name, api_key=os.getenv("GOOGLE_API_KEY"))

    else:
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY."
        )


# ── Document loading & chunking ───────────────────────────────────────────────

def load_chunks() -> tuple[list[list[str]], list[dict]]:
    """
    Load documents, split into chunks using the same strategy as VectorDB,
    and return:
        contexts        — List[List[str]]  for the Synthesizer
        chunk_registry  — List[dict]       for mapping context text → doc IDs
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    contexts:       list[list[str]] = []
    chunk_registry: list[dict]      = []

    for doc_idx, filename in enumerate(DOC_FILES):
        filepath = os.path.join(DATA_DIR, filename)

        if not os.path.exists(filepath):
            print(f"  ⚠ Skipping missing file: {filename}")
            continue

        with open(filepath, encoding='utf-8') as f:
            text = f.read()

        chunks = splitter.split_text(text)
        print(f"  doc_{doc_idx} ({filename}): {len(chunks)} chunks")

        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
            contexts.append([chunk])                    # Synthesizer wants List[List[str]]
            chunk_registry.append({
                'chunk_id': chunk_id,
                'text':     chunk,
            })

    return contexts, chunk_registry


# ── Relevant-ID resolution ────────────────────────────────────────────────────

def find_relevant_doc_ids(context_texts: list[str], chunk_registry: list[dict]) -> list[str]:
    """
    Map context chunk texts back to their chunk IDs.
    A chunk matches if the context text is a substring of the stored chunk text.
    """
    ids = []
    for ctx in context_texts:
        for entry in chunk_registry:
            if ctx.strip() and ctx.strip() in entry['text']:
                ids.append(entry['chunk_id'])
                break   # one context text → one chunk_id
    return ids


# ── Main synthesis pipeline ───────────────────────────────────────────────────

def synthesize_test_queries() -> list[dict]:
    """
    Full pipeline:
      1. Load & chunk documents
      2. Run Synthesizer.generate_goldens_from_contexts()
      3. Convert goldens → test_queries dicts with query / relevant_doc_ids / ground_truth
    """
    print("\n" + "=" * 60)
    print("DeepEval Synthesizer — Generating Test Queries")
    print("=" * 60)

    # ── Step 1: Load documents
    print("\n[1/3] Loading and chunking documents...")
    contexts, chunk_registry = load_chunks()
    print(f"      Total contexts: {len(contexts)}")

    # ── Step 2: Run synthesizer
    print("\n[2/3] Running synthesizer (this calls the LLM)...")
    model      = _init_model()
    synthesizer = Synthesizer(model=model)

    synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        include_expected_output=True,       # generates ground_truth
        max_goldens_per_context=MAX_GOLDENS_PER_CONTEXT,
    )

    goldens = synthesizer.synthetic_goldens
    print(f"      Generated {len(goldens)} goldens")

    # ── Step 3: Convert to RagEvaluator format
    print("\n[3/3] Mapping goldens → test_queries format...")
    test_queries: list[dict] = []

    for golden in goldens:
        query        = golden.input or ""
        ground_truth = golden.expected_output or ""
        context_texts = golden.context or []

        relevant_doc_ids = find_relevant_doc_ids(context_texts, chunk_registry)

        test_queries.append({
            'query':            query,
            'relevant_doc_ids': relevant_doc_ids,
            'ground_truth':     ground_truth,
        })

    return test_queries


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = synthesize_test_queries()

    # ── Print results
    print("\n" + "=" * 60)
    print(f"SYNTHESIZED TEST QUERIES ({len(test_queries)} total)")
    print("=" * 60)

    for i, tq in enumerate(test_queries, 1):
        print(f"\n[{i}] Query:            {tq['query']}")
        print(f"     relevant_doc_ids: {tq['relevant_doc_ids']}")
        print(f"     ground_truth:     {tq['ground_truth']}")

    # ── Save to JSON
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(test_queries, f, indent=2)

    print(f"\n✓ Saved to {OUTPUT_PATH}")
    print("\n── Drop into RagEvaluator ──────────────────────────────────")
    print("from tests.synthesize_test_queries import synthesize_test_queries")
    print("test_queries = synthesize_test_queries()")
    print("reporter.evaluate_rag_system(test_queries, n_results=3)")
    print("=" * 60)
