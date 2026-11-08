"""Evaluation metrics for TechQA RAG benchmark.

Implements two-stage evaluation:
1. Stage 1 (Retrieval): Evidence Retrieval Coverage (0/1) based on exact character offset intervals.
2. Stage 2 (Generation): Answer Factual Correctness (0/1) and detailed error diagnostics.
"""

from typing import List, Tuple, Dict, Any, Optional
import re
from src.retriever import RetrievedChunk


def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
  """Merges overlapping or contiguous half-open intervals [start, end)."""
  if not intervals:
    return []
  sorted_intervals = sorted(intervals, key=lambda x: x[0])
  merged = [sorted_intervals[0]]

  for current_start, current_end in sorted_intervals[1:]:
    last_start, last_end = merged[-1]
    if current_start <= last_end:
      merged[-1] = (last_start, max(last_end, current_end))
    else:
      merged.append((current_start, current_end))

  return merged


def evaluate_retrieval_coverage(
    delivered_chunks: List[RetrievedChunk],
    gold_doc_id: str,
    gold_start: int,
    gold_end: int,
) -> Dict[str, Any]:
  """Evaluates whether the union of delivered chunks from gold document covers the reference span [gold_start, gold_end).

  Returns:
      coverage: 1 if completely covered, else 0.
      gold_doc_retrieved: True if at least one chunk from gold_doc_id was delivered.
      delivered_gold_intervals: List of [start, end) intervals from gold document.
      covered_char_fraction: Fraction of reference characters covered.
  """
  gold_intervals = [
      (c.start_char, c.end_char)
      for c in delivered_chunks
      if c.doc_id == gold_doc_id
  ]

  gold_doc_retrieved = len(gold_intervals) > 0
  merged_gold = merge_intervals(gold_intervals)

  span_len = max(1, gold_end - gold_start)
  covered_chars = 0
  fully_covered = False

  for s, e in merged_gold:
    overlap_start = max(s, gold_start)
    overlap_end = min(e, gold_end)
    if overlap_end > overlap_start:
      covered_chars += overlap_end - overlap_start

    if s <= gold_start and e >= gold_end:
      fully_covered = True

  return {
      "coverage": 1 if fully_covered else 0,
      "gold_doc_retrieved": gold_doc_retrieved,
      "delivered_gold_intervals": gold_intervals,
      "covered_chars": covered_chars,
      "reference_len": span_len,
      "covered_fraction": min(1.0, covered_chars / span_len),
  }


def compute_token_f1(prediction: str, reference: str) -> float:
  """Computes token-level F1 score after standard punctuation and case normalization."""

  def normalize(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return s.split()

  pred_tokens = normalize(prediction)
  ref_tokens = normalize(reference)

  if not pred_tokens or not ref_tokens:
    return 1.0 if pred_tokens == ref_tokens else 0.0

  common = set(pred_tokens) & set(ref_tokens)
  num_same = sum(
      min(pred_tokens.count(tok), ref_tokens.count(tok)) for tok in common
  )

  if num_same == 0:
    return 0.0

  precision = num_same / len(pred_tokens)
  recall = num_same / len(ref_tokens)
  return (2 * precision * recall) / (precision + recall)


def evaluate_factual_correctness(
    generated_answer: str,
    gold_answer: str,
    question_text: str,
    context_text: str,
) -> Dict[str, Any]:
  """Evaluates factual correctness according to Assignment 3/4 Scoring Rubric (Appendix D).

  Correct = 1: Answers question with all essential reference facts and no materially false/contradictory claim.
  Correct = 0: Essential facts missing, materially incorrect, or abstention.
  """
  gen_clean = generated_answer.strip()
  gold_clean = gold_answer.strip()

  # 1. Check for abstention
  abstention_patterns = [
      r"information is not available",
      r"not mentioned in the provided",
      r"does not contain",
      r"cannot answer based on the provided",
      r"context does not provide",
  ]
  is_abstention = any(
      re.search(pat, gen_clean, re.IGNORECASE) for pat in abstention_patterns
  )

  if is_abstention:
    return {
        "is_correct": 0,
        "token_f1": 0.0,
        "term_recall": 0.0,
        "error_category": "ABSTENTION",
        "diagnostic": (
            "Model explicitly abstained claiming information not found in"
            " context."
        ),
    }

  # 2. Check for contradiction on boolean / support questions
  # e.g., if gold says "not supported" and gen says "Yes, ... supports"
  gold_says_no = bool(
      re.search(r"\bnot supported\b|\bno\b|\bcannot\b", gold_clean, re.I)
  )
  gen_says_yes = bool(
      re.search(r"^\s*yes\b|\bsupports\b|\bis supported\b", gen_clean, re.I)
  )
  if (
      gold_says_no
      and gen_says_yes
      and not re.search(r"\bnot supported\b", gen_clean, re.I)
  ):
    return {
        "is_correct": 0,
        "token_f1": compute_token_f1(gen_clean, gold_clean),
        "term_recall": 0.0,
        "error_category": "HALLUCINATED_AFFIRMATION",
        "diagnostic": (
            "Model asserted support/yes when gold fact specifies not"
            " supported."
        ),
    }

  # 3. Extract key phrases and technical terms
  tech_terms = set(
      re.findall(r"\b[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+\b", gold_clean)
      + re.findall(r"\b\d+\.\d+(?:\.\d+)*\b", gold_clean)
      + re.findall(r"\b[A-Z_]{3,}\b", gold_clean)
      + re.findall(r"(?:--|-)[a-zA-Z0-9_-]+", gold_clean)
  )

  token_f1 = compute_token_f1(gen_clean, gold_clean)

  if tech_terms:
    matched_terms = [t for t in tech_terms if t.lower() in gen_clean.lower()]
    term_recall = len(matched_terms) / len(tech_terms)
  else:
    term_recall = 1.0

  # Check if core reference phrases (4+ words) are quoted or reflected
  gold_phrases = [
      p.strip() for p in re.split(r"[,;.]", gold_clean) if len(p.split()) >= 3
  ]
  matched_phrases = [p for p in gold_phrases if p.lower() in gen_clean.lower()]
  has_core_phrase = len(matched_phrases) > 0

  # Decision rule:
  # - If core phrase from reference answer is matched and no direct contradiction
  # - Or token F1 >= 0.35 and term recall >= 0.50
  if has_core_phrase or (token_f1 >= 0.35 and term_recall >= 0.50):
    is_correct = 1
    category = "CORRECT"
    diagnostic = f"Factual agreement verified (Token F1: {token_f1:.2f}, Matched phrases: {len(matched_phrases)})."
  elif term_recall < 0.40 and token_f1 < 0.25:
    is_correct = 0
    category = "MISSING_ESSENTIAL_FACTS"
    diagnostic = (
        f"Critical parameters or facts missing (Tech term recall:"
        f" {term_recall:.2f}, Token F1: {token_f1:.2f})."
    )
  else:
    is_correct = 0
    category = "PARTIALLY_INCORRECT_OR_INCOMPLETE"
    diagnostic = (
        f"Incomplete facts or unverified assertions (Token F1: {token_f1:.2f})."
    )

  return {
      "is_correct": is_correct,
      "token_f1": token_f1,
      "term_recall": term_recall,
      "error_category": category,
      "diagnostic": diagnostic,
  }
