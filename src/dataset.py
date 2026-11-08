"""Dataset loader and validator for IBM TechQA.

Implements the official protocol from Assignment 3/4:
- Loads question-answering pairs and technote corpus.
- Concatenates QUESTION_TITLE + newline + QUESTION_TEXT.
- Validates answerability, candidate pool integrity (50 docs), and gold document membership.
- Verifies exact zero-based half-open char offset alignment: doc_text[start:end] == answer_text.
- Deterministic seeded sampling for frozen evaluation subsets.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random

from src.config import (
    TRAINING_QA_PATH,
    DEV_QA_PATH,
    TECHNOTES_PATH,
    CANDIDATES_PER_QUERY,
    RANDOM_SEED,
)


@dataclass
class TechQAQuestion:
  question_id: str
  title: str
  body: str
  full_query: str
  gold_doc_id: str
  candidate_doc_ids: List[str]
  answer_start: int
  answer_end: int
  gold_answer_text: str
  split: str


class TechQADataset:

  def __init__(
      self,
      technotes_path: Path = TECHNOTES_PATH,
      dev_qa_path: Path = DEV_QA_PATH,
      training_qa_path: Path = TRAINING_QA_PATH,
  ):
    self.technotes_path = technotes_path
    self.dev_qa_path = dev_qa_path
    self.training_qa_path = training_qa_path
    self.documents: Dict[str, str] = {}
    self.doc_titles: Dict[str, str] = {}
    self._loaded = False

  def load_corpus(self):
    """Loads technotes corpus plain text."""
    if self._loaded:
      return
    with open(self.technotes_path, "r", encoding="utf-8") as f:
      raw_docs = json.load(f)

    for doc_id, doc_data in raw_docs.items():
      # Use the preprocessed 'text' field which matches exact character offsets
      self.documents[doc_id] = doc_data.get("text", "")
      self.doc_titles[doc_id] = doc_data.get("title", "")
    self._loaded = True

  def get_document_text(self, doc_id: str) -> str:
    if not self._loaded:
      self.load_corpus()
    return self.documents.get(doc_id, "")

  def load_and_filter_questions(
      self, split: str = "dev"
  ) -> Tuple[List[TechQAQuestion], Dict[str, int]]:
    """Loads QA pairs and filters strictly according to Assignment protocol.

    Rejection criteria:
    1. Unanswerable (ANSWERABLE != 'Y')
    2. Gold document missing from technotes corpus
    3. Candidate pool does not contain exactly CANDIDATES_PER_QUERY documents
    4. Gold document not present in candidate pool
    5. Span mismatch: doc_text[start:end] != reference answer
    """
    if not self._loaded:
      self.load_corpus()

    target_path = self.dev_qa_path if split == "dev" else self.training_qa_path
    with open(target_path, "r", encoding="utf-8") as f:
      raw_qas = json.load(f)

    eligible = []
    stats = {
        "total": len(raw_qas),
        "unanswerable": 0,
        "missing_gold_doc": 0,
        "gold_not_in_candidates": 0,
        "invalid_pool_size": 0,
        "span_mismatch": 0,
        "eligible": 0,
    }

    seen_queries = set()

    for item in raw_qas:
      qid = item.get("QUESTION_ID")
      if item.get("ANSWERABLE") != "Y":
        stats["unanswerable"] += 1
        continue

      gold_doc = item.get("DOCUMENT")
      if not gold_doc or gold_doc not in self.documents:
        stats["missing_gold_doc"] += 1
        continue

      candidates = item.get("DOC_IDS", [])
      if len(candidates) != CANDIDATES_PER_QUERY:
        stats["invalid_pool_size"] += 1
        continue

      if gold_doc not in candidates:
        stats["gold_not_in_candidates"] += 1
        continue

      try:
        start_offset = int(item.get("START_OFFSET"))
        end_offset = int(item.get("END_OFFSET"))
        gold_ans = item.get("ANSWER", "")

        doc_slice = self.documents[gold_doc][start_offset:end_offset]
        if doc_slice != gold_ans:
          stats["span_mismatch"] += 1
          continue
      except (ValueError, TypeError, KeyError):
        stats["span_mismatch"] += 1
        continue

      title = item.get("QUESTION_TITLE", "").strip()
      body = item.get("QUESTION_TEXT", "").strip()
      full_query = f"{title}\n{body}".strip()

      if full_query in seen_queries:
        continue
      seen_queries.add(full_query)

      q_obj = TechQAQuestion(
          question_id=qid,
          title=title,
          body=body,
          full_query=full_query,
          gold_doc_id=gold_doc,
          candidate_doc_ids=candidates,
          answer_start=start_offset,
          answer_end=end_offset,
          gold_answer_text=gold_ans,
          split=split,
      )
      eligible.append(q_obj)

    stats["eligible"] = len(eligible)
    return eligible, stats

  def get_evaluation_subset(
      self, n_samples: int = 50, seed: int = RANDOM_SEED
  ) -> List[TechQAQuestion]:
    """Deterministically selects n_samples from eligible dev set using seed."""
    eligible_dev, _ = self.load_and_filter_questions(split="dev")
    # Sort deterministically by question_id before shuffling
    eligible_dev.sort(key=lambda x: x.question_id)
    rng = random.Random(seed)
    rng.shuffle(eligible_dev)
    return eligible_dev[:n_samples]

  def get_pilot_questions(self) -> List[TechQAQuestion]:
    """Returns the two pilot questions from Assignment 3: TRAIN_Q046 and TRAIN_Q050."""
    eligible_train, _ = self.load_and_filter_questions(split="train")
    pilot_ids = {"TRAIN_Q046", "TRAIN_Q050"}
    return [q for q in eligible_train if q.question_id in pilot_ids]
