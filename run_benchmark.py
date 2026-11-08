"""Main execution script for Assignment 4 benchmark: Fixed vs Semantic Chunking in TechQA.

Supports:
- --pilot: Exactly reproduces and extends Assignment 3 pilot (TRAIN_Q046, TRAIN_Q050) with Qwen generation.
- Full evaluation run: Evaluates 50 frozen questions from TechQA dev set.
- Generates runs.jsonl (Pick-a-number ready), summary.json, and manifest.json.
"""

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch

from src.config import (
    FIXED_CHUNK_SIZE,
    SEMANTIC_TARGET_MIN,
    SEMANTIC_HARD_MAX,
    CONTEXT_TOKEN_BUDGET,
    TOP_K_CHUNKS,
    ENCODER_MODEL_ID,
    ENCODER_REVISION,
    GENERATOR_MODEL_ID,
    GENERATOR_REVISION,
    RANDOM_SEED,
    OUTPUTS_DIR,
)
from src.chunkers import fixed_chunks, semantic_chunks
from src.dataset import TechQADataset, TechQAQuestion
from src.retriever import EncoderBackend, VectorRetriever
from src.generator import QwenGenerator
from src.metrics import evaluate_retrieval_coverage, evaluate_factual_correctness
from src.stats import (
    compute_mcnemar_exact,
    compute_paired_bootstrap_ci,
    holm_bonferroni_correction,
    summarize_continuous_metric,
)


def get_file_sha256(path: Path) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    while chunk := f.read(8192):
      h.update(chunk)
  return h.hexdigest()


def process_question_condition(
    q: TechQAQuestion,
    condition: str,
    dataset: TechQADataset,
    backend: EncoderBackend,
    generator: QwenGenerator,
) -> Dict[str, Any]:
  """Executes retrieval and generation for a single question under a given chunking condition."""
  t0 = time.perf_counter()

  # 1. Chunk all candidate documents
  all_chunks = []
  doc_chunk_counts = {}

  for doc_id in q.candidate_doc_ids:
    doc_text = dataset.get_document_text(doc_id)
    if not doc_text:
      continue

    if condition == "fixed_token":
      spans = fixed_chunks(doc_text, backend, FIXED_CHUNK_SIZE)
    elif condition == "semantic":
      spans = semantic_chunks(
          doc_text, backend, SEMANTIC_TARGET_MIN, SEMANTIC_HARD_MAX
      )
    else:
      raise ValueError(f"Unknown condition: {condition}")

    doc_chunk_counts[doc_id] = len(spans)
    for idx, (s, e) in enumerate(spans):
      chunk_text = doc_text[s:e]
      all_chunks.append({
          "chunk_id": f"{doc_id}_c{idx:03d}",
          "doc_id": doc_id,
          "start_char": s,
          "end_char": e,
          "content": chunk_text,
      })

  total_corpus_chunks = len(all_chunks)

  # 2. Vector indexing & retrieval
  retriever = VectorRetriever(backend, use_faiss=True)
  retriever.build_index(all_chunks)
  top_chunks = retriever.retrieve(q.full_query, top_k=TOP_K_CHUNKS)

  # 3. Context packing under 1280 Qwen tokens budget
  packed_context, delivered_chunks, qwen_tokens, is_truncated = (
      generator.pack_context(top_chunks, budget_tokens=CONTEXT_TOKEN_BUDGET)
  )

  # 4. Retrieval coverage evaluation (char offsets)
  cov_res = evaluate_retrieval_coverage(
      delivered_chunks, q.gold_doc_id, q.answer_start, q.answer_end
  )

  # 5. Answer generation
  gen_answer, gen_tokens = generator.generate_answer(
      question_title=q.title,
      question_body=q.body,
      packed_context=packed_context,
  )

  # 6. Factual correctness evaluation
  fact_res = evaluate_factual_correctness(
      generated_answer=gen_answer,
      gold_answer=q.gold_answer_text,
      question_text=q.full_query,
      context_text=packed_context,
  )

  elapsed_ms = (time.perf_counter() - t0) * 1000.0

  # Structured run record (Pick-a-number ready)
  run_record = {
      "question_id": q.question_id,
      "split": q.split,
      "condition": condition,
      "full_query": q.full_query,
      "gold_doc_id": q.gold_doc_id,
      "gold_start": q.answer_start,
      "gold_end": q.answer_end,
      "gold_answer": q.gold_answer_text,
      "total_corpus_chunks": total_corpus_chunks,
      "top_chunks": [
          {
              "chunk_id": c.chunk_id,
              "doc_id": c.doc_id,
              "start_char": c.start_char,
              "end_char": c.end_char,
              "score": round(c.score, 4),
              "rank": c.rank,
              "text_snippet": c.content[:150] + "...",
          }
          for c in top_chunks
      ],
      "delivered_chunk_ids": [c.chunk_id for c in delivered_chunks],
      "delivered_gold_intervals": cov_res["delivered_gold_intervals"],
      "evidence_coverage": cov_res["coverage"],
      "gold_doc_retrieved": cov_res["gold_doc_retrieved"],
      "covered_fraction": round(cov_res["covered_fraction"], 4),
      "qwen_context_tokens": qwen_tokens,
      "context_truncated": is_truncated,
      "packed_context": packed_context,
      "generated_answer": gen_answer,
      "generated_tokens": gen_tokens,
      "factual_correctness": fact_res["is_correct"],
      "token_f1": round(fact_res["token_f1"], 4),
      "term_recall": round(fact_res.get("term_recall", 0.0), 4),
      "error_category": fact_res["error_category"],
      "diagnostic": fact_res["diagnostic"],
      "elapsed_ms": round(elapsed_ms, 2),
  }

  return run_record


def run_benchmark(
    n_samples: int = 50,
    is_pilot: bool = False,
    out_dir: Path = OUTPUTS_DIR,
    seed: int = RANDOM_SEED,
):
  out_dir.mkdir(parents=True, exist_ok=True)
  runs_file = out_dir / "runs.jsonl"
  summary_file = out_dir / "summary.json"
  manifest_file = out_dir / "manifest.json"

  print("Initializing dataset...")
  dataset = TechQADataset()
  dataset.load_corpus()

  if is_pilot:
    questions = dataset.get_pilot_questions()
    print(f"Loaded {len(questions)} Assignment 3 Pilot questions.")
  else:
    questions = dataset.get_evaluation_subset(n_samples=n_samples, seed=seed)
    print(f"Loaded {len(questions)} Frozen Evaluation questions (seed={seed}).")

  print("Initializing encoder and generator...")
  backend = EncoderBackend()
  generator = QwenGenerator()

  runs_records = []
  # Clear existing runs file if fresh run
  if runs_file.exists():
    runs_file.unlink()

  print("\n" + "=" * 70)
  print(
      f"STARTING EXPERIMENT: {len(questions)} Questions x 2 Conditions (Fixed vs"
      " Semantic)"
  )
  print("=" * 70)

  fixed_coverages = []
  semantic_coverages = []
  fixed_correctnesses = []
  semantic_correctnesses = []

  fixed_tokens = []
  semantic_tokens = []
  fixed_chunks_counts = []
  semantic_chunks_counts = []
  fixed_latencies = []
  semantic_latencies = []

  for idx, q in enumerate(questions, start=1):
    print(
        f"\n[{idx}/{len(questions)}] Processing {q.question_id}:"
        f" '{q.title[:60]}...'"
    )

    # 1. Fixed Condition
    rec_fixed = process_question_condition(
        q, "fixed_token", dataset, backend, generator
    )
    with open(runs_file, "a", encoding="utf-8") as f:
      f.write(json.dumps(rec_fixed, ensure_ascii=False) + "\n")
      f.flush()

    # 2. Semantic Condition
    rec_sem = process_question_condition(
        q, "semantic", dataset, backend, generator
    )
    with open(runs_file, "a", encoding="utf-8") as f:
      f.write(json.dumps(rec_sem, ensure_ascii=False) + "\n")
      f.flush()

    runs_records.extend([rec_fixed, rec_sem])

    # Collect stats
    fixed_coverages.append(rec_fixed["evidence_coverage"])
    semantic_coverages.append(rec_sem["evidence_coverage"])
    fixed_correctnesses.append(rec_fixed["factual_correctness"])
    semantic_correctnesses.append(rec_sem["factual_correctness"])

    fixed_tokens.append(rec_fixed["qwen_context_tokens"])
    semantic_tokens.append(rec_sem["qwen_context_tokens"])
    fixed_chunks_counts.append(rec_fixed["total_corpus_chunks"])
    semantic_chunks_counts.append(rec_sem["total_corpus_chunks"])
    fixed_latencies.append(rec_fixed["elapsed_ms"])
    semantic_latencies.append(rec_sem["elapsed_ms"])

    print(
        f"  Fixed:    Coverage={rec_fixed['evidence_coverage']}, Correct={rec_fixed['factual_correctness']}, Tokens={rec_fixed['qwen_context_tokens']}, Chunks={rec_fixed['total_corpus_chunks']}"
    )
    print(
        f"  Semantic: Coverage={rec_sem['evidence_coverage']}, Correct={rec_sem['factual_correctness']}, Tokens={rec_sem['qwen_context_tokens']}, Chunks={rec_sem['total_corpus_chunks']}"
    )

  # --- Statistical Computations ---
  print("\n" + "=" * 70)
  print("COMPUTING STATISTICAL TESTS AND AGGREGATE METRICS...")
  print("=" * 70)

  # H1: Evidence Retrieval Coverage
  mcnemar_cov = compute_mcnemar_exact(fixed_coverages, semantic_coverages)
  boot_cov = compute_paired_bootstrap_ci(
      fixed_coverages, semantic_coverages, seed=seed
  )

  # H2: Answer Factual Correctness
  mcnemar_corr = compute_mcnemar_exact(
      fixed_correctnesses, semantic_correctnesses
  )
  boot_corr = compute_paired_bootstrap_ci(
      fixed_correctnesses, semantic_correctnesses, seed=seed
  )

  # Holm correction for H1 and H2
  holm_res = holm_bonferroni_correction({
      "H1_Evidence_Coverage": mcnemar_cov["p_value"],
      "H2_Factual_Correctness": mcnemar_corr["p_value"],
  })

  # Continuous metrics
  tokens_summary = summarize_continuous_metric(fixed_tokens, semantic_tokens)
  chunks_summary = summarize_continuous_metric(
      fixed_chunks_counts, semantic_chunks_counts
  )
  latency_summary = summarize_continuous_metric(
      fixed_latencies, semantic_latencies
  )

  # Objective 3: Association between Evidence Coverage and Correctness
  # Cross-tabulate within Fixed and Semantic
  def assoc_table(covs, corrs):
    c1_a1 = sum(1 for c, a in zip(covs, corrs) if c == 1 and a == 1)
    c1_a0 = sum(1 for c, a in zip(covs, corrs) if c == 1 and a == 0)
    c0_a1 = sum(1 for c, a in zip(covs, corrs) if c == 0 and a == 1)
    c0_a0 = sum(1 for c, a in zip(covs, corrs) if c == 0 and a == 0)
    rate_given_cov = c1_a1 / (c1_a1 + c1_a0) if (c1_a1 + c1_a0) > 0 else 0.0
    rate_given_nocov = c0_a1 / (c0_a1 + c0_a0) if (c0_a1 + c0_a0) > 0 else 0.0
    return {
        "cov1_corr1": c1_a1,
        "cov1_corr0": c1_a0,
        "cov0_corr1": c0_a1,
        "cov0_corr0": c0_a0,
        "correctness_when_covered": round(rate_given_cov, 4),
        "correctness_when_not_covered": round(rate_given_nocov, 4),
    }

  assoc_fixed = assoc_table(fixed_coverages, fixed_correctnesses)
  assoc_sem = assoc_table(semantic_coverages, semantic_correctnesses)

  summary = {
      "experiment_metadata": {
          "n_questions": len(questions),
          "is_pilot": is_pilot,
          "random_seed": seed,
          "encoder_model": f"{ENCODER_MODEL_ID} ({ENCODER_REVISION})",
          "generator_model": f"{GENERATOR_MODEL_ID} ({GENERATOR_REVISION})",
          "budget_limit_qwen_tokens": CONTEXT_TOKEN_BUDGET,
          "top_k_chunks": TOP_K_CHUNKS,
      },
      "H1_Evidence_Retrieval_Coverage": {
          "fixed_rate": round(mcnemar_cov["rate_baseline"], 4),
          "semantic_rate": round(mcnemar_cov["rate_proposed"], 4),
          "absolute_difference": round(mcnemar_cov["absolute_difference"], 4),
          "mcnemar_p_value": mcnemar_cov["p_value"],
          "mcnemar_contingency": mcnemar_cov["contingency_table"],
          "bootstrap_95_ci": [
              round(boot_cov["ci_lower"], 4),
              round(boot_cov["ci_upper"], 4),
          ],
          "holm_adjusted": holm_res["H1_Evidence_Coverage"],
      },
      "H2_Answer_Factual_Correctness": {
          "fixed_rate": round(mcnemar_corr["rate_baseline"], 4),
          "semantic_rate": round(mcnemar_corr["rate_proposed"], 4),
          "absolute_difference": round(mcnemar_corr["absolute_difference"], 4),
          "mcnemar_p_value": mcnemar_corr["p_value"],
          "mcnemar_contingency": mcnemar_corr["contingency_table"],
          "bootstrap_95_ci": [
              round(boot_corr["ci_lower"], 4),
              round(boot_corr["ci_upper"], 4),
          ],
          "holm_adjusted": holm_res["H2_Factual_Correctness"],
      },
      "Objective_3_Coverage_Correctness_Association": {
          "fixed_condition": assoc_fixed,
          "semantic_condition": assoc_sem,
      },
      "Implementation_Diagnostics": {
          "qwen_context_tokens": tokens_summary,
          "corpus_chunk_counts": chunks_summary,
          "latency_ms": latency_summary,
      },
  }

  with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

  # Manifest file for complete reproducibility
  manifest = {
      "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
      "system": {
          "os": platform.platform(),
          "python": platform.python_version(),
          "torch": torch.__version__,
          "cuda_available": torch.cuda.is_available(),
          "device_name": (
              torch.cuda.get_device_name(0)
              if torch.cuda.is_available()
              else "CPU"
          ),
      },
      "models": {
          "encoder": {"id": ENCODER_MODEL_ID, "revision": ENCODER_REVISION},
          "generator": {
              "id": GENERATOR_MODEL_ID,
              "revision": GENERATOR_REVISION,
          },
      },
      "parameters": {
          "fixed_chunk_size": FIXED_CHUNK_SIZE,
          "semantic_min": SEMANTIC_TARGET_MIN,
          "semantic_max": SEMANTIC_HARD_MAX,
          "context_budget_qwen": CONTEXT_TOKEN_BUDGET,
          "top_k": TOP_K_CHUNKS,
          "seed": seed,
      },
      "code_hashes": {
          "chunkers.py": get_file_sha256(
              Path(__file__).parent / "src" / "chunkers.py"
          ),
          "retriever.py": get_file_sha256(
              Path(__file__).parent / "src" / "retriever.py"
          ),
          "generator.py": get_file_sha256(
              Path(__file__).parent / "src" / "generator.py"
          ),
          "metrics.py": get_file_sha256(
              Path(__file__).parent / "src" / "metrics.py"
          ),
          "stats.py": get_file_sha256(
              Path(__file__).parent / "src" / "stats.py"
          ),
      },
  }

  with open(manifest_file, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

  print("\n" + "=" * 70)
  print(f"BENCHMARK COMPLETED SUCCESSFULLY!")
  print(f"Raw Logs:  {runs_file}")
  print(f"Summary:   {summary_file}")
  print(f"Manifest:  {manifest_file}")
  print("=" * 70)

  print(
      "\n=== SUMMARY: EVIDENCE COVERAGE (H1) ==="
  )
  print(f"Fixed:    {summary['H1_Evidence_Retrieval_Coverage']['fixed_rate']*100:.1f}%")
  print(f"Semantic: {summary['H1_Evidence_Retrieval_Coverage']['semantic_rate']*100:.1f}%")
  print(f"Diff:     {summary['H1_Evidence_Retrieval_Coverage']['absolute_difference']*100:+.1f}% (95% CI: [{summary['H1_Evidence_Retrieval_Coverage']['bootstrap_95_ci'][0]*100:.1f}%, {summary['H1_Evidence_Retrieval_Coverage']['bootstrap_95_ci'][1]*100:.1f}%])")
  print(f"McNemar p-value: {summary['H1_Evidence_Retrieval_Coverage']['mcnemar_p_value']:.4f}")

  print("\n=== SUMMARY: FACTUAL CORRECTNESS (H2) ===")
  print(f"Fixed:    {summary['H2_Answer_Factual_Correctness']['fixed_rate']*100:.1f}%")
  print(f"Semantic: {summary['H2_Answer_Factual_Correctness']['semantic_rate']*100:.1f}%")
  print(f"Diff:     {summary['H2_Answer_Factual_Correctness']['absolute_difference']*100:+.1f}% (95% CI: [{summary['H2_Answer_Factual_Correctness']['bootstrap_95_ci'][0]*100:.1f}%, {summary['H2_Answer_Factual_Correctness']['bootstrap_95_ci'][1]*100:.1f}%])")
  print(f"McNemar p-value: {summary['H2_Answer_Factual_Correctness']['mcnemar_p_value']:.4f}")

  return summary


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Run TechQA Chunking Benchmark for Assignment 4"
  )
  parser.add_argument(
      "--n-samples",
      type=int,
      default=50,
      help="Number of evaluation questions",
  )
  parser.add_argument(
      "--pilot", action="store_true", help="Run on Assignment 3 pilot questions"
  )
  parser.add_argument(
      "--seed", type=int, default=RANDOM_SEED, help="Random seed"
  )
  parser.add_argument(
      "--out",
      type=str,
      default=str(OUTPUTS_DIR),
      help="Output directory path",
  )
  args = parser.parse_args()

  run_benchmark(
      n_samples=args.n_samples,
      is_pilot=args.pilot,
      out_dir=Path(args.out),
      seed=args.seed,
  )
