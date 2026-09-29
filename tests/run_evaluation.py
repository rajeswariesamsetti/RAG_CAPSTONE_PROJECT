"""
Evaluation script for the RAG Knowledge Assistant.
Executes the test suite defined in tests/evaluation_questions.json and calculates actual metrics.
Does NOT fabricate accuracy numbers.
"""

import json
import os
import sys
from pathlib import Path

# Enforce UTF-8 on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import (
    DATA_DIR,
    TESTS_DIR,
    DEFAULT_SIMILARITY_TOP_K,
    EMBEDDING_DIM,
    OUT_OF_CONTEXT_RESPONSE,
    GROQ_API_KEY
)
from src.document_loader import load_documents
from src.chunking import chunk_documents
from src.vector_store import build_vector_index, load_vector_index, get_collection_stats
from src.rag_pipeline import RAGPipeline
from src.retriever import retrieve_chunks
from utils.logging_utils import get_logger

logger = get_logger("evaluation")

def run_evaluation(top_k: int = DEFAULT_SIMILARITY_TOP_K):
    """
    Executes the comprehensive evaluation against evaluation_questions.json.
    """
    questions_file = TESTS_DIR / "evaluation_questions.json"
    if not questions_file.exists():
        print(f"Error: {questions_file} not found.")
        return

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print("\n" + "=" * 75)
    print("RAG KNOWLEDGE ASSISTANT - CAPSTONE EVALUATION BENCHMARK")
    print("=" * 75)

    # Step 1: Ensure Knowledge Base is indexed
    sample_dir = DATA_DIR / "sample_documents"
    docs_to_load = sample_dir if sample_dir.exists() and list(sample_dir.glob("*.txt")) else DATA_DIR

    print(f"\n[1/4] Preparing Knowledge Base from: {docs_to_load}")
    docs = load_documents(docs_to_load)
    if not docs:
        print("[!] No documents found to evaluate. Please populate data/sample_documents/ first.")
        return

    nodes = chunk_documents(docs)
    index = build_vector_index(nodes)
    stats = get_collection_stats()

    print(f"  [+] Documents Loaded:   {len(docs)}")
    print(f"  [+] Chunks Generated:   {len(nodes)}")
    print(f"  [+] Vectors in Chroma:  {stats['chunk_count']}")
    print(f"  [+] Embedding Model:    sentence-transformers/all-MiniLM-L6-v2")
    print(f"  [+] Embedding Dim:      {EMBEDDING_DIM}")
    print(f"  [+] Retrieval Top-K:    {top_k}")

    # Step 2: Initialize RAG Pipeline
    has_api_key = bool(os.getenv("GROQ_API_KEY", GROQ_API_KEY).strip())
    pipeline = None
    if has_api_key:
        print("\n[2/4] Groq API Key detected. Full LLM generation enabled.")
        try:
            pipeline = RAGPipeline(index=index, similarity_top_k=top_k)
        except Exception as e:
            print(f"[!] Warning: Could not initialize Groq LLM ({e}). Falling back to retrieval metrics.")
            pipeline = None
    else:
        print("\n[2/4] GROQ_API_KEY not configured. Running Retrieval & Grounding Analysis.")

    # Step 3: Run queries
    print(f"\n[3/4] Evaluating {len(questions)} Benchmark Questions...")
    print("-" * 75)
    print(f"{'ID':<3} | {'Type':<17} | {'Retrieved Top Source':<22} | {'Score':<6} | {'Status'}")
    print("-" * 75)

    results = []
    correct_retrievals = 0
    total_in_domain = 0
    correctly_rejected_outside = 0
    total_outside = 0
    all_scores = []

    for q in questions:
        q_id = q["id"]
        q_text = q["question"]
        q_type = q["type"]
        expected_src = q.get("expected_source")

        # Perform retrieval
        retrieved = retrieve_chunks(q_text, index, similarity_top_k=top_k)
        top_chunk = retrieved[0] if retrieved else None
        top_score = top_chunk["score"] if top_chunk else 0.0
        top_file = top_chunk["filename"] if top_chunk else "None"

        if top_score is not None:
            all_scores.append(top_score)

        llm_answer = None
        status = "PASS"

        if q_type in ("direct", "multi_chunk"):
            total_in_domain += 1
            # Check if expected source is among retrieved sources (by filename or document stem)
            retrieved_files = [c["filename"] for c in retrieved]
            retrieved_stems = [Path(c["filename"]).stem.lower() for c in retrieved]
            expected_stem = Path(expected_src).stem.lower() if expected_src else ""
            if expected_src and (expected_src in retrieved_files or expected_stem in retrieved_stems):
                correct_retrievals += 1
                status = "[+] RETRIEVED"
            else:
                status = "[-] MISSED"

            if pipeline:
                try:
                    res = pipeline.answer_question(q_text)
                    llm_answer = res["answer"]
                except Exception as e:
                    llm_answer = f"Error: {e}"

        elif q_type == "outside_knowledge":
            total_outside += 1
            if pipeline:
                try:
                    res = pipeline.answer_question(q_text)
                    llm_answer = res["answer"]
                    if llm_answer == OUT_OF_CONTEXT_RESPONSE:
                        correctly_rejected_outside += 1
                        status = "[+] REJECTED (SAFE)"
                    else:
                        status = "[-] HALLUCINATED"
                except Exception:
                    status = "LLM ERROR"
            else:
                status = "OUT-OF-DOMAIN"

        print(f"{q_id:<3} | {q_type:<17} | {top_file[:22]:<22} | {str(top_score):<6} | {status}")

        results.append({
            "id": q_id,
            "question": q_text,
            "type": q_type,
            "expected_source": expected_src,
            "retrieved_sources": [c["filename"] for c in retrieved],
            "top_score": top_score,
            "status": status,
            "llm_answer": llm_answer
        })

    # Step 4: Summary Metrics
    print("\n" + "=" * 75)
    print("EVALUATION METRICS REPORT")
    print("=" * 75)
    print(f"Total Evaluated Questions:             {len(questions)}")
    print(f"  - Direct Questions:                  {sum(1 for q in questions if q['type'] == 'direct')}")
    print(f"  - Multi-Chunk Questions:             {sum(1 for q in questions if q['type'] == 'multi_chunk')}")
    print(f"  - Outside-Knowledge Questions:       {total_outside}")
    print("-" * 75)
    hit_rate = round((correct_retrievals/total_in_domain)*100, 2) if total_in_domain else 0.0
    print(f"In-Domain Target Document Hits:        {correct_retrievals} / {total_in_domain} ({hit_rate}%)")

    if all_scores:
        avg_score = round(sum(all_scores) / len(all_scores), 4)
        min_score = round(min(all_scores), 4)
        max_score = round(max(all_scores), 4)
        print(f"Retrieval Cosine Similarities:         Min={min_score}, Max={max_score}, Avg={avg_score}")

    if pipeline:
        rej_rate = round((correctly_rejected_outside/total_outside)*100, 2) if total_outside else 0.0
        print(f"Outside-Knowledge Correct Rejections: {correctly_rejected_outside} / {total_outside} ({rej_rate}%)")
    else:
        print("Outside-Knowledge LLM Rejections:    (Requires GROQ_API_KEY in .env)")

    print("=" * 75)

    # Save results to file
    out_file = TESTS_DIR / "evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "total_questions": len(questions),
                "in_domain_total": total_in_domain,
                "in_domain_hits": correct_retrievals,
                "outside_knowledge_total": total_outside,
                "outside_knowledge_rejected": correctly_rejected_outside if pipeline else None,
                "average_similarity_score": round(sum(all_scores) / len(all_scores), 4) if all_scores else None
            },
            "detailed_results": results
        }, f, indent=2)
    print(f"\nSaved detailed evaluation metrics to: {out_file}\n")

if __name__ == "__main__":
    run_evaluation()
