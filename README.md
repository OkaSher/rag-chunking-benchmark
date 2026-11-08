# Fixed Token and Semantic Chunking in Technical Support Question Answering
### Research Methods and Tools — Assignment 4: Experiment Execution, Results and Discussion
**Authors:** Ramazan Nagimov, Nurgeldi Yessengeldi, Zhetpis Baigulov  
**Release Tag:** `assignment-4`  
**Dataset:** IBM TechQA (CDLA-Permissive 1.0)  
**Hardware Verified:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, CUDA 12.4, PyTorch 2.6.0+cu124)

---

## 1. Project Overview

This repository contains the complete, reproducible research pipeline comparing **Fixed-Token Chunking** (256-token baseline, zero overlap) against **Adjacent-Sentence Semantic Chunking** (80th percentile cosine distance threshold, [128, 320] tokens) for technical support question answering on the IBM TechQA benchmark.

The benchmark executes a two-stage Retrieval-Augmented Generation (RAG) architecture:
1. **Stage 1 (Retrieval):** BM25 pre-filtered candidate pool (50 IBM technotes per query, ~200–300 chunks) + exact in-memory inner product dense retrieval (`faiss.IndexFlatIP`) using `sentence-transformers/all-mpnet-base-v2` (768-dim, unit-normalized).
2. **Stage 2 (Generation):** Rank-ordered context packing under a strict 1280-token budget (Qwen tokenizer) + deterministic greedy answer generation via `Qwen/Qwen2.5-1.5B-Instruct` (`temperature=0.0`, `max_new_tokens=256`).

---

## 2. Main Experimental Findings ($N = 20$ Questions, 40 Runs)

| Metric | Fixed Baseline | Semantic (Proposed) | Difference ($\Delta$ [95% CI]) | Statistical Test & Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Evidence Retrieval Coverage ($H_1$)** | 40.0% (8/20) | 35.0% (7/20) | -5.0% [-20%, +10%] | Exact McNemar $p = 1.0000$ (Not Significant) |
| **Answer Factual Correctness ($H_2$)** | 10.0% (2/20) | 15.0% (3/20) | +5.0% [-10%, +20%] | Exact McNemar $p = 1.0000$ (Not Significant) |
| **Correctness \| Evidence Covered** | 12.5% (1/8) | 28.6% (2/7) | +16.1% | Descriptive (2.3× higher fidelity with semantic chunks) |
| **Correctness \| Evidence Missed** | 8.3% (1/12) | 7.7% (1/13) | -0.6% | Baseline prior guess rate |
| **Mean End-to-end Latency** | 9.47 s | 48.13 s | +38.65 s (5.1× slower) | Wilcoxon $p < 0.0001^{***}$ (Qu et al. 2025 trade-off) |

---

## 3. Repository Structure

```
rag-chunking-benchmark/
├── src/
│   ├── config.py             # Pinned model revisions, chunking parameters, random seed
│   ├── chunkers.py           # Fixed-token and semantic chunkers with WordPiece retreat
│   ├── dataset.py            # TechQA validation, offset verification, frozen sampling
│   ├── retriever.py          # VectorRetriever with FAISS IndexFlatIP & MPNet embeddings
│   ├── generator.py          # QwenGenerator with 1280 context packing & greedy decoding
│   ├── metrics.py            # Evidence Coverage (0/1) & Factual Correctness scoring
│   └── stats.py              # Exact McNemar test, 95% Bootstrap CI, Holm-Bonferroni
├── outputs/assignment_4/
│   ├── runs.jsonl            # Raw per-query execution logs (40 runs, Pick-a-number)
│   ├── summary.json          # Consolidated statistical metrics and test p-values
│   ├── manifest.json         # SHA-256 code hashes, model revisions, environment info
│   ├── error_analysis_table.md # 37 categorized failure cases
│   ├── figure1_retrieval_and_correctness_comparison.png # 300 DPI publication figure
│   └── figure2_diagnostics_and_tradeoffs.png            # 300 DPI publication figure
├── run_benchmark.py          # Batch runner executing the full 40 runs
├── compute_final_results.py  # Aggregator script computing metrics and tests
├── analyze_errors.py         # Error analysis taxonomy classifier
├── generate_plots.py         # Matplotlib publication figures generator
├── build_full_assignment4_docs.py # Generates Assignment4_Nagimov_Yessengeldi_Baigulov.docx
├── DEFENCE_GUIDE.md          # Complete oral defence guide for the 5 examination questions
├── requirements.txt          # Frozen library versions
└── README.md                 # This documentation
```

---

## 4. Setup and Reproduction Instructions

### Step 1: Environment Setup
Ensure Python 3.10+ (tested on Python 3.12.10 with PyTorch 2.6.0+cu124):
```bash
pip install -r requirements.txt
```
*Note for CUDA acceleration:* Install PyTorch matching your CUDA version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Step 2: Reproduce the Full 40-Run Benchmark
To run the frozen evaluation set of 20 development questions (40 evaluations):
```bash
python run_benchmark.py --split dev --limit 20 --seed 42
```
This streams results line-by-line into `outputs/assignment_4/runs.jsonl`.

### Step 3: Compute Summary Statistics
```bash
python compute_final_results.py
```
This regenerates `outputs/assignment_4/summary.json` containing exact McNemar tests, 95% bootstrap CIs, Wilcoxon tests, and Holm-Bonferroni thresholds.

### Step 4: Generate Publication Figures
```bash
python generate_plots.py
```
This creates high-resolution PNGs:
- `figure1_retrieval_and_correctness_comparison.png`
- `figure2_diagnostics_and_tradeoffs.png`

### Step 5: Perform Error Analysis
```bash
python analyze_errors.py
```
This categorizes all failure cases into `outputs/assignment_4/error_analysis_table.md` (37 cases).

### Step 6: Generate Assignment 4 Final Report (.docx and .md)
```bash
python build_full_assignment4_docs.py
python generate_markdown_report.py
```

---

## 5. "Pick-a-Number" Verification for Oral Defence

Under the course grading rules, every single reported number must be traceable to a specific log line:
- **Metrics in Table 2:** Open `outputs/assignment_4/summary.json` to verify `fixed_rate: 0.4`, `semantic_rate: 0.35`, `mcnemar_p_value: 1.0`, etc.
- **Per-question runs:** Open `outputs/assignment_4/runs.jsonl`. For example, Line 1 (`DEV_Q271`, fixed) shows:
  ```json
  "question_id": "DEV_Q271",
  "condition": "fixed_token",
  "evidence_coverage": 1,
  "factual_correctness": 1,
  "qwen_context_tokens": 1056,
  "elapsed_ms": 10427
  ```
- **Error taxonomy:** Open `outputs/assignment_4/error_analysis_table.md` to see individual cases for each error mode.

---

## 6. Citation & License
- IBM TechQA is released under the **CDLA-Permissive 1.0** license.
- Code is released under the **MIT License**.
