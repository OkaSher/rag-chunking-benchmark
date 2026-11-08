"""Chunking algorithms for Fixed-Token and Semantic Segmentation.

Implements exact algorithms specified in Assignment 3 (Appendix C) and optimized
for Assignment 4 with strict character offset preservation.
"""

import math
import re
from typing import List, Tuple, Any
import numpy as np


def sentence_units(text: str) -> List[Tuple[int, int]]:
  """Splits text into conservative sentence or line units with exact character offsets.

  Preserves all characters from 0 to len(text) without gaps or offset drift.
  Respects sentence terminals (.!?), paragraph boundaries, and list/code
  markers.
  """
  if not text:
    return []

  pattern = re.compile(
      r'(?:(?<=[.!?])\s+(?=[A-Z0-9"\'`])|\n{2,}|\n(?=[*#-]\s|[A-Z0-9]))'
  )

  units = []
  last = 0
  for m in pattern.finditer(text):
    end = m.end()
    if end > last:
      units.append((last, end))
      last = end

  if last < len(text):
    units.append((last, len(text)))

  return units if units else [(0, len(text))]


def fixed_chunks(text: str, tokenizer: Any, size: int) -> List[Tuple[int, int]]:
  """Half-open source intervals [start, end); include whitespace without offset drift.

  Matches the verbatim specification from Assignment 3 Appendix C.
  Uses tokenizer offset mappings to map token boundaries back to character
  indices,
  safeguarding against WordPiece token count expansion upon slicing.
  """
  if size <= 0:
    raise ValueError("Chunk size must be positive")
  if not text:
    return []

  spans = []
  start = 0
  text_len = len(text)

  while start < text_len:
    offsets = tokenizer.offsets(text[start:])
    if not offsets:
      if spans:
        spans[-1] = (spans[-1][0], text_len)
      else:
        spans.append((0, text_len))
      break

    cut = min(size, len(offsets))
    end = text_len if cut == len(offsets) else start + offsets[cut][0]

    # Safeguard against WordPiece expansion upon slicing
    while end > start and tokenizer.count(text[start:end]) > size:
      cut -= 1
      if cut <= 0:
        break
      end = start + offsets[cut][0]

    if end <= start:
      end = (
          start + offsets[1][0]
          if len(offsets) > 1
          else min(start + 50, text_len)
      )

    spans.append((start, end))
    start = end

  return spans


def semantic_chunks(
    text: str, backend: Any, minimum: int = 128, maximum: int = 320
) -> List[Tuple[int, int]]:
  """Adjacent-sentence semantic chunking with token-count bounds.

  Matches verbatim specification from Assignment 3 Appendix C.
  1. Breaks text into sentence/line units.
  2. Sub-segments any unit exceeding maximum token cap using fixed_chunks.
  3. Computes unit embeddings and adjacent cosine distances.
  4. Sets distance threshold at 80th percentile.
  5. Places boundaries when target minimum is reached AND distance > threshold,
     or when hard maximum token cap is exceeded.
  """
  if not (0 < minimum <= maximum):
    raise ValueError("Require 0 < minimum <= maximum")
  if not text:
    return []

  raw_units = sentence_units(text)
  if not raw_units:
    return []

  raw_texts = [text[a:z] for a, z in raw_units]
  if hasattr(backend, "count_batch"):
    raw_counts = backend.count_batch(raw_texts)
  else:
    raw_counts = [backend.count(t) for t in raw_texts]

  units = []
  for (a, z), cnt in zip(raw_units, raw_counts):
    if cnt > maximum:
      sub_spans = fixed_chunks(text[a:z], backend, maximum)
      units.extend((a + i, a + j) for i, j in sub_spans)
    else:
      units.append((a, z))

  if not units:
    return []
  if len(units) == 1:
    return units

  # Embed units and calculate cosine distance between adjacent units
  unit_texts = [text[a:z] for a, z in units]
  vectors = backend.encode(unit_texts)

  # Normalize vectors for fast cosine similarity
  norms = np.linalg.norm(vectors, axis=1, keepdims=True)
  norms[norms == 0] = 1.0
  normed_vectors = vectors / norms

  # Dot products of adjacent pairs
  sims = np.sum(normed_vectors[:-1] * normed_vectors[1:], axis=1)
  sims = np.clip(sims, -1.0, 1.0)
  distances = 1.0 - sims

  threshold = (
      float(np.percentile(distances, 80.0))
      if len(distances) > 0
      else float("inf")
  )

  # Pre-calculate unit token counts using fast batch tokenization
  if hasattr(backend, "count_batch"):
    unit_tokens = backend.count_batch(unit_texts)
  else:
    unit_tokens = [backend.count(u_text) for u_text in unit_texts]

  result = []
  start, end = units[0][0], units[0][1]
  accumulated_tokens = unit_tokens[0]

  for i, (a, z) in enumerate(units[1:]):
    next_tokens = unit_tokens[i + 1]
    force = (accumulated_tokens + next_tokens) > maximum
    change = accumulated_tokens >= minimum and distances[i] > threshold

    if force or change:
      result.append((start, end))
      start = a
      accumulated_tokens = next_tokens
    else:
      accumulated_tokens += next_tokens

    end = z

  result.append((start, end))
  return result
