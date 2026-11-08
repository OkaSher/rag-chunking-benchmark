"""Finalizes the 20-question evaluation run:
- Computes McNemar tests, 95% bootstrap CIs, and Holm-Bonferroni corrections.
- Computes Objective 3 cross-tabulation.
- Generates summary.json and manifest.json.
- Runs error analysis (>=30 cases) and generates publication figures.
"""

import hashlib
import json
import platform
import time
from pathlib import Path
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
from src.stats import (
    compute_mcnemar_exact,
    compute_paired_bootstrap_ci,
    holm_bonferroni_correction,
    summarize_continuous_metric,
)
from analyze_errors import run_error_analysis
from generate_plots import generate_figures


def get_file_sha256(path: Path) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    while chunk := f.read(8192):
      h.update(chunk)
  return h.hexdigest()


def compute_and_save():
  runs_file = OUTPUTS_DIR / "runs.jsonl"
  summary_file = OUTPUTS_DIR / "summary.json"
  manifest_file = OUTPUTS_DIR / "manifest.json"

  with open(runs_file, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

  q_ids = sorted(list(set(r["question_id"] for r in records)))
  n_questions = len(q_ids)
  print(f"Loaded {len(records)} runs for {n_questions} questions.")

  fixed_recs = {
      r["question_id"]: r for r in records if r["condition"] == "fixed_token"
  }
  sem_recs = {
      r["question_id"]: r for r in records if r["condition"] == "semantic"
  }

  fixed_covs = [fixed_recs[qid]["evidence_coverage"] for qid in q_ids]
  sem_covs = [sem_recs[qid]["evidence_coverage"] for qid in q_ids]

  fixed_corrs = [fixed_recs[qid]["factual_correctness"] for qid in q_ids]
  sem_corrs = [sem_recs[qid]["factual_correctness"] for qid in q_ids]

  fixed_toks = [fixed_recs[qid]["qwen_context_tokens"] for qid in q_ids]
  sem_toks = [sem_recs[qid]["qwen_context_tokens"] for qid in q_ids]

  fixed_chunks = [fixed_recs[qid]["total_corpus_chunks"] for qid in q_ids]
  sem_chunks = [sem_recs[qid]["total_corpus_chunks"] for qid in q_ids]

  fixed_lats = [fixed_recs[qid]["elapsed_ms"] for qid in q_ids]
  sem_lats = [sem_recs[qid]["elapsed_ms"] for qid in q_ids]

  # 1. H1: Evidence Retrieval Coverage
  mcnemar_cov = compute_mcnemar_exact(fixed_covs, sem_covs)
  boot_cov = compute_paired_bootstrap_ci(
      fixed_covs, sem_covs, seed=RANDOM_SEED
  )

  # 2. H2: Answer Factual Correctness
  mcnemar_corr = compute_mcnemar_exact(fixed_corrs, sem_corrs)
  boot_corr = compute_paired_bootstrap_ci(
      fixed_corrs, sem_corrs, seed=RANDOM_SEED
  )

  # 3. Holm correction
  holm_res = holm_bonferroni_correction({
      "H1_Evidence_Coverage": mcnemar_cov["p_value"],
      "H2_Factual_Correctness": mcnemar_corr["p_value"],
  })

  # 4. Continuous diagnostics
  tokens_diag = summarize_continuous_metric(fixed_toks, sem_toks)
  chunks_diag = summarize_continuous_metric(fixed_chunks, sem_chunks)
  latency_diag = summarize_continuous_metric(fixed_lats, sem_lats)

  # 5. Objective 3: Cross-tabulation of Coverage vs Correctness
  def cross_tab(covs, corrs):
    c1_a1 = sum(1 for c, a in zip(covs, corrs) if c == 1 and a == 1)
    c1_a0 = sum(1 for c, a in zip(covs, corrs) if c == 1 and a == 0)
    c0_a1 = sum(1 for c, a in zip(covs, corrs) if c == 0 and a == 1)
    c0_a0 = sum(1 for c, a in zip(covs, corrs) if c == 0 and a == 0)
    n_cov = c1_a1 + c1_a0
    n_nocov = c0_a1 + c0_a0
    rate_cov = c1_a1 / n_cov if n_cov > 0 else 0.0
    rate_nocov = c0_a1 / n_nocov if n_nocov > 0 else 0.0
    return {
        "covered_total": n_cov,
        "covered_correct": c1_a1,
        "covered_incorrect": c1_a0,
        "correctness_rate_when_covered": round(rate_cov, 4),
        "not_covered_total": n_nocov,
        "not_covered_correct": c0_a1,
        "not_covered_incorrect": c0_a0,
        "correctness_rate_when_not_covered": round(rate_nocov, 4),
    }

  obj3_fixed = cross_tab(fixed_covs, fixed_corrs)
  obj3_sem = cross_tab(sem_covs, sem_corrs)

  summary = {
      "experiment_metadata": {
          "dataset": "IBM TechQA (CDLA-Permissive 1.0)",
          "split": "dev (frozen evaluation subset)",
          "n_questions": n_questions,
          "total_evaluations": len(records),
          "random_seed": RANDOM_SEED,
          "encoder_model": f"{ENCODER_MODEL_ID} ({ENCODER_REVISION})",
          "generator_model": f"{GENERATOR_MODEL_ID} ({GENERATOR_REVISION})",
          "context_budget_qwen_tokens": CONTEXT_TOKEN_BUDGET,
          "top_k_chunks": TOP_K_CHUNKS,
      },
      "H1_Evidence_Retrieval_Coverage": {
          "fixed_rate": round(mcnemar_cov["rate_baseline"], 4),
          "semantic_rate": round(mcnemar_cov["rate_proposed"], 4),
          "absolute_difference": round(mcnemar_cov["absolute_difference"], 4),
          "mcnemar_contingency": mcnemar_cov["contingency_table"],
          "mcnemar_statistic": mcnemar_cov["mcnemar_statistic"],
          "mcnemar_p_value": round(mcnemar_cov["p_value"], 4),
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
          "mcnemar_contingency": mcnemar_corr["contingency_table"],
          "mcnemar_statistic": mcnemar_corr["mcnemar_statistic"],
          "mcnemar_p_value": round(mcnemar_corr["p_value"], 4),
          "bootstrap_95_ci": [
              round(boot_corr["ci_lower"], 4),
              round(boot_corr["ci_upper"], 4),
          ],
          "holm_adjusted": holm_res["H2_Factual_Correctness"],
      },
      "Objective_3_Coverage_Correctness_Association": {
          "fixed_condition": obj3_fixed,
          "semantic_condition": obj3_sem,
      },
      "Implementation_Diagnostics": {
          "qwen_context_tokens": tokens_diag,
          "corpus_chunk_counts": chunks_diag,
          "latency_ms": latency_diag,
      },
  }

  with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

  # Manifest
  manifest = {
      "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
      "hardware": {
          "device": (
              torch.cuda.get_device_name(0)
              if torch.cuda.is_available()
              else "CPU"
          ),
          "os": platform.platform(),
          "python": platform.python_version(),
          "torch": torch.__version__,
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
          "seed": RANDOM_SEED,
          "n_questions": n_questions,
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

  print("Saved summary.json and manifest.json.")

  # Error analysis
  print("\nRunning Error Analysis...")
  err_stats = run_error_analysis(runs_file, OUTPUTS_DIR, min_cases=30)

  # Generate figures
  print("\nGenerating publication figures...")
  generate_figures(summary_file, runs_file, OUTPUTS_DIR)

  print("\n=== FINAL RESULTS SUMMARY ===")
  print(
      f"Evidence Coverage (H1): Fixed = {summary['H1_Evidence_Retrieval_Coverage']['fixed_rate']*100:.1f}%, Semantic = {summary['H1_Evidence_Retrieval_Coverage']['semantic_rate']*100:.1f}%, Diff = {summary['H1_Evidence_Retrieval_Coverage']['absolute_difference']*100:+.1f}% (p = {summary['H1_Evidence_Retrieval_Coverage']['mcnemar_p_value']:.4f})"
  )
  print(
      f"Factual Correctness (H2): Fixed = {summary['H2_Answer_Factual_Correctness']['fixed_rate']*100:.1f}%, Semantic = {summary['H2_Answer_Factual_Correctness']['semantic_rate']*100:.1f}%, Diff = {summary['H2_Answer_Factual_Correctness']['absolute_difference']*100:+.1f}% (p = {summary['H2_Answer_Factual_Correctness']['mcnemar_p_value']:.4f})"
  )
  print(
      f"Context Tokens: Fixed = {summary['Implementation_Diagnostics']['qwen_context_tokens']['baseline']['mean']:.1f} tokens, Semantic = {summary['Implementation_Diagnostics']['qwen_context_tokens']['proposed']['mean']:.1f} tokens"
  )
  print(f"Error cases catalogued: {err_stats['total_errors']}")


if __name__ == "__main__":
  compute_and_save()
