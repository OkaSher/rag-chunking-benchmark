"""Plot generation module for Assignment 4 (Part 3 Figures).

Generates two publication-ready figures:
1. Figure 1: Evidence Coverage & Answer Correctness Comparison (with 95% Bootstrap CI).
2. Figure 2: System Diagnostics & Granularity Trade-offs (Token Budget Utilization & Chunk Counts).
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import OUTPUTS_DIR


def generate_figures(
    summary_path: Path = OUTPUTS_DIR / "summary.json",
    runs_path: Path = OUTPUTS_DIR / "runs.jsonl",
    out_dir: Path = OUTPUTS_DIR,
):
  if not summary_path.exists() or not runs_path.exists():
    print(f"Summary or runs log not found in {out_dir}. Skipping figure generation.")
    return

  with open(summary_path, "r", encoding="utf-8") as f:
    summary = json.load(f)

  records = []
  with open(runs_path, "r", encoding="utf-8") as f:
    for line in f:
      if line.strip():
        records.append(json.loads(line.strip()))

  # Set styling
  sns.set_theme(style="whitegrid", font="sans-serif")
  palette = {"fixed_token": "#4C72B0", "semantic": "#55A868"}

  # ---------------------------------------------------------------------------
  # FIGURE 1: Primary Hypothesis Outcomes (Coverage & Correctness) with 95% CI
  # ---------------------------------------------------------------------------
  fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

  h1_fixed = summary["H1_Evidence_Retrieval_Coverage"]["fixed_rate"] * 100.0
  h1_sem = summary["H1_Evidence_Retrieval_Coverage"]["semantic_rate"] * 100.0
  h1_ci = [
      v * 100.0
      for v in summary["H1_Evidence_Retrieval_Coverage"]["bootstrap_95_ci"]
  ]

  h2_fixed = summary["H2_Answer_Factual_Correctness"]["fixed_rate"] * 100.0
  h2_sem = summary["H2_Answer_Factual_Correctness"]["semantic_rate"] * 100.0
  h2_ci = [
      v * 100.0
      for v in summary["H2_Answer_Factual_Correctness"]["bootstrap_95_ci"]
  ]

  labels = [
      "Evidence Retrieval Coverage (H1)",
      "Answer Factual Correctness (H2)",
  ]
  fixed_vals = [h1_fixed, h2_fixed]
  sem_vals = [h1_sem, h2_sem]

  x = np.arange(len(labels))
  width = 0.32

  rects1 = ax.bar(
      x - width / 2,
      fixed_vals,
      width,
      label="Fixed-Token Baseline (256 cap)",
      color=palette["fixed_token"],
      edgecolor="black",
      linewidth=1.0,
      alpha=0.9,
  )
  rects2 = ax.bar(
      x + width / 2,
      sem_vals,
      width,
      label="Adjacent Semantic (128-320 tokens)",
      color=palette["semantic"],
      edgecolor="black",
      linewidth=1.0,
      alpha=0.9,
  )

  # Add error bars on semantic representing paired bootstrap CI relative to baseline
  h1_p = summary["H1_Evidence_Retrieval_Coverage"]["mcnemar_p_value"]
  h2_p = summary["H2_Answer_Factual_Correctness"]["mcnemar_p_value"]

  # Annotations on top of bars
  for r in rects1:
    h = r.get_height()
    ax.annotate(
        f"{h:.1f}%",
        xy=(r.get_x() + r.get_width() / 2, h),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
  for r in rects2:
    h = r.get_height()
    ax.annotate(
        f"{h:.1f}%",
        xy=(r.get_x() + r.get_width() / 2, h),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

  # Annotate difference and p-values
  diff_h1 = summary["H1_Evidence_Retrieval_Coverage"]["absolute_difference"] * 100.0
  diff_h2 = summary["H2_Answer_Factual_Correctness"]["absolute_difference"] * 100.0
  ax.text(
      0,
      max(h1_fixed, h1_sem) + 12,
      f"Δ = {diff_h1:+.1f}%\n95% CI: [{h1_ci[0]:.1f}%, {h1_ci[1]:.1f}%]\nMcNemar"
      f" p = {h1_p:.4f}",
      ha="center",
      va="center",
      fontsize=8.5,
      bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="#b0bec5"),
  )
  ax.text(
      1,
      max(h2_fixed, h2_sem) + 12,
      f"Δ = {diff_h2:+.1f}%\n95% CI: [{h2_ci[0]:.1f}%, {h2_ci[1]:.1f}%]\nMcNemar"
      f" p = {h2_p:.4f}",
      ha="center",
      va="center",
      fontsize=8.5,
      bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="#b0bec5"),
  )

  ax.set_ylabel("Rate (%)", fontsize=11, fontweight="bold")
  ax.set_title(
      "Figure 1: Comparison of Fixed-Token vs. Semantic Chunking in TechQA"
      " RAG",
      fontsize=12,
      fontweight="bold",
      pad=15,
  )
  ax.set_xticks(x)
  ax.set_xticklabels(labels, fontsize=10.5, fontweight="bold")
  ax.set_ylim(0, 105)
  ax.legend(loc="upper right", frameon=True, fontsize=9.5)
  plt.tight_layout()

  fig1_path = out_dir / "figure1_retrieval_and_correctness_comparison.png"
  fig.savefig(fig1_path, dpi=300)
  plt.close(fig)
  print(f"Saved Figure 1 to {fig1_path}")

  # ---------------------------------------------------------------------------
  # FIGURE 2: System Diagnostics (Context Budget & Chunk Count Distributions)
  # ---------------------------------------------------------------------------
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

  fixed_tok = [
      r["qwen_context_tokens"]
      for r in records
      if r["condition"] == "fixed_token"
  ]
  sem_tok = [
      r["qwen_context_tokens"] for r in records if r["condition"] == "semantic"
  ]

  # Subplot 1: Delivered context tokens
  ax1.boxplot(
      [fixed_tok, sem_tok],
      tick_labels=["Fixed-Token", "Semantic"],
      patch_artist=True,
      boxprops=dict(facecolor="#e1f5fe", color="#0277bd"),
      medianprops=dict(color="#d32f2f", linewidth=2),
  )
  ax1.axhline(
      1280,
      color="#c62828",
      linestyle="--",
      linewidth=1.2,
      label="Budget Limit (1280 tokens)",
  )
  ax1.set_ylabel("Qwen Token Count", fontsize=10, fontweight="bold")
  ax1.set_title(
      "(a) Delivered Context Token Budget Utilization",
      fontsize=11,
      fontweight="bold",
  )
  ax1.legend(loc="lower right", fontsize=8.5)

  # Subplot 2: Total Chunks generated per candidate pool (50 docs)
  fixed_c = [
      r["total_corpus_chunks"]
      for r in records
      if r["condition"] == "fixed_token"
  ]
  sem_c = [
      r["total_corpus_chunks"] for r in records if r["condition"] == "semantic"
  ]

  ax2.boxplot(
      [fixed_c, sem_c],
      tick_labels=["Fixed-Token", "Semantic"],
      patch_artist=True,
      boxprops=dict(facecolor="#e8f5e9", color="#2e7d32"),
      medianprops=dict(color="#d32f2f", linewidth=2),
  )
  ax2.set_ylabel("Corpus Chunks (50 docs)", fontsize=10, fontweight="bold")
  ax2.set_title(
      "(b) Segmentation Granularity per Query", fontsize=11, fontweight="bold"
  )

  plt.suptitle(
      "Figure 2: Implementation Diagnostics and Packing Trade-offs",
      fontsize=12,
      fontweight="bold",
      y=1.02,
  )
  plt.tight_layout()

  fig2_path = out_dir / "figure2_diagnostics_and_tradeoffs.png"
  fig.savefig(fig2_path, dpi=300)
  plt.close(fig)
  print(f"Saved Figure 2 to {fig2_path}")


if __name__ == "__main__":
  generate_figures()
