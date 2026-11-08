"""Answer Generator and Context Packing using Qwen2.5-1.5B-Instruct.

Addresses Professor's Remark #2:
- Implements the complete Generation stage on top of retrieved context.
- Packs top-k chunks into a strict context budget (<= 1280 Qwen tokens) including separators.
- Generates answer using greedy decoding (temperature=0.0) for reproducible scientific comparison.
"""

from typing import List, Tuple, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    GENERATOR_MODEL_ID,
    GENERATOR_REVISION,
    CONTEXT_TOKEN_BUDGET,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    DO_SAMPLE,
)
from src.retriever import RetrievedChunk


class QwenGenerator:

  def __init__(
      self,
      model_id: str = GENERATOR_MODEL_ID,
      revision: str = GENERATOR_REVISION,
      device: Optional[str] = None,
  ):
    if device is None:
      self.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
      self.device = device

    print(
        f"Loading generator model '{model_id}' ({revision}) on {self.device}..."
    )
    self.tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, use_fast=True
    )

    dtype = torch.float16 if self.device == "cuda" else torch.float32
    self.model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        device_map=self.device,
    )
    self.model.eval()

  def count_tokens(self, text: str) -> int:
    """Counts tokens using the Qwen tokenizer."""
    if not text:
      return 0
    return len(self.tokenizer.encode(text, add_special_tokens=False))

  def pack_context(
      self,
      chunks: List[RetrievedChunk],
      budget_tokens: int = CONTEXT_TOKEN_BUDGET,
  ) -> Tuple[str, List[RetrievedChunk], int, bool]:
    """Packs top chunks in rank order within the Qwen tokenizer budget.

    Returns:
        packed_text: Formatted context string.
        delivered_chunks: List of chunks that fit in budget.
        token_count: Actual token count of packed context.
        truncated: Boolean indicating if any top-k chunk had to be skipped.
    """
    delivered = []
    current_tokens = 0
    truncated = False

    for chunk in chunks:
      chunk_header = (
          f"[Document: {chunk.doc_id} (Rank {chunk.rank}, Score:"
          f" {chunk.score:.4f})]\n"
      )
      chunk_body = f"{chunk.content.strip()}\n\n"
      formatted_chunk = chunk_header + chunk_body
      chunk_tokens = self.count_tokens(formatted_chunk)

      if current_tokens + chunk_tokens <= budget_tokens:
        delivered.append((formatted_chunk, chunk))
        current_tokens += chunk_tokens
      else:
        truncated = True

    packed_text = "".join([item[0] for item in delivered]).strip()
    delivered_chunks = [item[1] for item in delivered]
    final_tokens = self.count_tokens(packed_text)

    return packed_text, delivered_chunks, final_tokens, truncated

  def generate_answer(
      self,
      question_title: str,
      question_body: str,
      packed_context: str,
  ) -> Tuple[str, int]:
    """Formats prompt and generates technical answer with Qwen2.5-1.5B-Instruct.

    Returns:
        generated_answer: Text of generated response.
        gen_tokens: Number of generated tokens.
    """
    system_prompt = (
        "You are an expert IBM technical support assistant. "
        "Answer the user's technical question accurately and concisely using ONLY the provided technical documentation context. "
        "Include specific commands, parameters, configuration settings, or troubleshooting steps when present. "
        "If the provided context does not contain the answer, state 'The information is not available in the provided documents.'"
    )

    user_query = f"{question_title}\n{question_body}".strip()
    user_content = (
        "Technical Documentation Context:\n"
        "--------------------------------\n"
        f"{packed_context if packed_context else 'No context retrieved.'}\n"
        "--------------------------------\n\n"
        f"Question:\n{user_query}\n\n"
        "Please provide the exact technical answer based on the documentation above:"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    prompt_text = self.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

    with torch.no_grad():
      output_ids = self.model.generate(
          **inputs,
          max_new_tokens=MAX_NEW_TOKENS,
          do_sample=DO_SAMPLE,
          temperature=None,
          top_p=None,
          pad_token_id=self.tokenizer.eos_token_id,
      )

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[0, prompt_len:]
    answer = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    gen_tokens = len(gen_ids)

    return answer, gen_tokens
