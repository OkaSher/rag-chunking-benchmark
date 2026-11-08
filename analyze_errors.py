"""Error Analysis module for Assignment 4 (Part 4 and Appendix B).

Extracts and categorizes >= 30 failure cases according to error taxonomies:
1. RETRIEVAL_GOLD_DOC_MISSED: Gold document was not ranked in top-k chunks.
2. RETRIEVAL_BOUNDARY_SPLIT: Gold document retrieved, but chunking cut the answer span.
3. BUDGET_EXCLUDED: Gold chunk was retrieved in top-k but skipped by the 1280-token cap.
4. GENERATION_OMISSION: Gold evidence was retrieved, but generator omitted essential parameters.
5. GENERATION_ABSTENTION: Gold evidence was present, but generator falsely abstained.
6. GENERATION_HALLUCINATION: Model added unsupported technical claims.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from src.config import OUTPUTS_DIR


def run_error_analysis(
    runs_path: Path = OUTPUTS_DIR / "runs.jsonl",
    out_dir: Path = OUTPUTS_DIR,
    min_cases: int = 30,
) -> Dict[str, Any]:
  if not runs_path.exists():
    raise FileNotFoundError(f"Runs log not found at: {runs_path}")

  records = []
  with open(runs_path, "r", encoding="utf-8") as f:
    for line in f:
      if line.strip():
        records.append(json.loads(line.strip()))

  print(f"Loaded {len(records)} run records from {runs_path}")

  # Identify errors: coverage == 0 or factual_correctness == 0
  error_cases = []

  for r in records:
    cov = r.get("evidence_coverage", 0)
    corr = r.get("factual_correctness", 0)

    if cov == 1 and corr == 1:
      continue  # Success case

    gold_doc = r.get("gold_doc_id", "")
    gold_retrieved = r.get("gold_doc_retrieved", False)
    covered_frac = r.get("covered_fraction", 0.0)
    truncated = r.get("context_truncated", False)
    gen_ans = r.get("generated_answer", "")
    err_cat = r.get("error_category", "")

    # Classify specific taxonomy
    if not gold_retrieved:
      taxonomy = "RETRIEVAL_GOLD_DOC_MISSED"
      explanation = f"Retriever failed to rank any chunk from gold document '{gold_doc}' into top-5."
      mitigation = (
          "Hybrid BM25 + dense retrieval or cross-encoder re-ranking."
      )
    elif cov == 0 and covered_frac > 0.0:
      taxonomy = "RETRIEVAL_BOUNDARY_SPLIT"
      explanation = f"Gold document was retrieved, but chunk boundary truncated reference span (covered {covered_frac*100:.1f}%)."
      mitigation = "Overlapping chunks or sentence-merging window."
    elif cov == 0 and truncated:
      taxonomy = "RETRIEVAL_BUDGET_EXCLUSION"
      explanation = "Gold chunk was in top-k, but skipped because earlier chunks exhausted the 1280-token budget."
      mitigation = "Dynamic context budget compression or extractive summary."
    elif err_cat == "ABSTENTION":
      taxonomy = "GENERATION_FALSE_ABSTENTION"
      explanation = "Generator falsely claimed information was absent despite evidence in context."
      mitigation = (
          "Refine prompt instructions to encourage confidence on partial"
          " matches."
      )
    elif err_cat == "MISSING_ESSENTIAL_FACTS":
      taxonomy = "GENERATION_FACTUAL_OMISSION"
      explanation = "Generator answered generally but omitted key commands, parameters, or version flags."
      mitigation = (
          "Few-shot prompt formatting with explicit technical entity checklist."
      )
    else:
      taxonomy = "GENERATION_UNSUPPORTED_OR_INCORRECT"
      explanation = (
          "Generator produced technical assertions that diverged from the"
          " reference fact."
      )
      mitigation = "Constrained decoding or citation enforcement."

    error_cases.append({
        "question_id": r.get("question_id"),
        "condition": r.get("condition"),
        "query": r.get("full_query")[:200] + "...",
        "gold_doc_id": gold_doc,
        "gold_answer": r.get("gold_answer"),
        "evidence_coverage": cov,
        "covered_fraction": covered_frac,
        "factual_correctness": corr,
        "generated_answer": gen_ans[:250] + "...",
        "error_taxonomy": taxonomy,
        "explanation": explanation,
        "mitigation": mitigation,
    })

  print(f"Total error cases found: {len(error_cases)}")

  # Taxonomy frequency breakdown
  breakdown = {}
  for e in error_cases:
    tax = e["error_taxonomy"]
    breakdown[tax] = breakdown.get(tax, 0) + 1

  # Group by condition
  breakdown_by_condition = {
      "fixed_token": {},
      "semantic": {},
  }
  for e in error_cases:
    cond = e["condition"]
    tax = e["error_taxonomy"]
    breakdown_by_condition[cond][tax] = (
        breakdown_by_condition[cond].get(tax, 0) + 1
    )

  # Select at least min_cases (or all) for report Appendix B
  selected_cases = error_cases[: max(min_cases, len(error_cases))]

  # Save JSONL
  out_jsonl = out_dir / "error_analysis_30_cases.jsonl"
  with open(out_jsonl, "w", encoding="utf-8") as f:
    for c in selected_cases:
      f.write(json.dumps(c, ensure_ascii=False) + "\n")

  # Generate Markdown table for Appendix B
  out_md = out_dir / "error_analysis_table.md"
  with open(out_md, "w", encoding="utf-8") as f:
    f.write("# Appendix B: Error Analysis Table (Assignment 4)\n\n")
    f.write("### Error Taxonomy Distribution\n\n")
    f.write(
        "| Error Category | Fixed-Token Count | Semantic Count | Total |"
        " Primary Mitigation |\n"
    )
    f.write(
        "| :--- | :---: | :---: | :---: | :--- |\n"
    )
    all_taxes = sorted(list(breakdown.keys()))
    for tax in all_taxes:
      fixed_cnt = breakdown_by_condition["fixed_token"].get(tax, 0)
      sem_cnt = breakdown_by_condition["semantic"].get(tax, 0)
      total_cnt = breakdown[tax]
      # sample mitigation
      sample_mit = next(
          c["mitigation"] for c in error_cases if c["error_taxonomy"] == tax
      )
      f.write(
          f"| `{tax}` | {fixed_cnt} | {sem_cnt} | {total_cnt} | {sample_mit}"
          " |\n"
      )

    f.write(f"\n### Detailed Case Manifest ({len(selected_cases)} Cases)\n\n")
    f.write(
        "| Case # | Question ID | Condition | Taxonomy | Golden Document &"
        " Answer Span | Generated Answer Snippet | Error Diagnostic |\n"
    )
    f.write(
        "| :---: | :--- | :---: | :--- | :--- | :--- | :--- |\n"
    )
    for idx, c in enumerate(selected_cases[:35], start=1):
      q_snippet = c["question_id"]
      gold_snip = f"**{c['gold_doc_id']}**: {c['gold_answer'][:60]}..."
      gen_snip = c["generated_answer"][:60].replace("\n", " ") + "..."
      f.write(
          f"| {idx} | `{q_snippet}` | {c['condition']} | `{c['error_taxonomy']}`"
          f" | {gold_snip} | {gen_snip} | {c['explanation']} |\n"
      )

  print(f"Saved error analysis to {out_jsonl} and {out_md}")
  return {
      "total_errors": len(error_cases),
      "selected_count": len(selected_cases),
      "breakdown": breakdown,
      "breakdown_by_condition": breakdown_by_condition,
  }


if __name__ == "__main__":
  run_error_analysis()
