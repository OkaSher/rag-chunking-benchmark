"""Statistical tests and confidence intervals for paired experimental comparison.

Implements requirements from Assignment 4 (Table 2) and Assignment 3 (Section 7):
- Exact two-sided McNemar test on discordant pairs (statsmodels / scipy binomial).
- 95% paired bootstrap confidence intervals (B=10,000 resamples, seed=42).
- Holm-Bonferroni familywise error correction for H1 and H2.
- Paired Wilcoxon signed-rank tests for continuous secondary metrics (tokens, latency).
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

from src.config import BOOTSTRAP_ROUNDS, RANDOM_SEED, ALPHA


def compute_mcnemar_exact(
    y_baseline: List[int], y_proposed: List[int]
) -> Dict[str, Any]:
  """Computes 2x2 contingency table and two-sided exact McNemar test on paired binary observations."""
  assert len(y_baseline) == len(
      y_proposed
  ), "Baseline and proposed lists must have equal length"
  n = len(y_baseline)

  # Contingency counts
  # a: Both 1
  # b: Baseline 1, Proposed 0
  # c: Baseline 0, Proposed 1
  # d: Both 0
  a = sum(1 for b, p in zip(y_baseline, y_proposed) if b == 1 and p == 1)
  b = sum(1 for b, p in zip(y_baseline, y_proposed) if b == 1 and p == 0)
  c = sum(1 for b, p in zip(y_baseline, y_proposed) if b == 0 and p == 1)
  d = sum(1 for b, p in zip(y_baseline, y_proposed) if b == 0 and p == 0)

  table = [[a, b], [c, d]]

  rate_baseline = (a + b) / n if n > 0 else 0.0
  rate_proposed = (a + c) / n if n > 0 else 0.0
  diff = rate_proposed - rate_baseline

  # Exact binomial test on discordant pairs b and c
  discordant = b + c
  if discordant == 0:
    p_value = 1.0
    stat = 0.0
  else:
    # Two-sided exact test
    res = mcnemar(table, exact=True)
    p_value = float(res.pvalue)
    stat = float(res.statistic)

  return {
      "n_total": n,
      "contingency_table": {"both_1": a, "base1_prop0": b, "base0_prop1": c, "both_0": d},
      "rate_baseline": rate_baseline,
      "rate_proposed": rate_proposed,
      "absolute_difference": diff,
      "discordant_count": discordant,
      "mcnemar_statistic": stat,
      "p_value": p_value,
      "significant_unadjusted": p_value < ALPHA,
  }


def compute_paired_bootstrap_ci(
    y_baseline: List[int],
    y_proposed: List[int],
    n_bootstraps: int = BOOTSTRAP_ROUNDS,
    seed: int = RANDOM_SEED,
    ci_level: float = 0.95,
) -> Dict[str, Any]:
  """Computes paired bootstrap confidence interval for the rate difference (Proposed - Baseline).

  Resamples whole question pairs with replacement.
  """
  assert len(y_baseline) == len(y_proposed)
  n = len(y_baseline)
  b_arr = np.array(y_baseline)
  p_arr = np.array(y_proposed)

  rng = np.random.default_rng(seed)
  diffs = np.empty(n_bootstraps)

  for i in range(n_bootstraps):
    indices = rng.integers(0, n, size=n)
    sample_b = b_arr[indices]
    sample_p = p_arr[indices]
    diffs[i] = np.mean(sample_p) - np.mean(sample_b)

  lower_pct = ((1.0 - ci_level) / 2.0) * 100.0
  upper_pct = (1.0 - (1.0 - ci_level) / 2.0) * 100.0

  ci_lower = float(np.percentile(diffs, lower_pct))
  ci_upper = float(np.percentile(diffs, upper_pct))
  mean_diff = float(np.mean(diffs))

  return {
      "observed_diff": float(np.mean(p_arr) - np.mean(b_arr)),
      "bootstrap_mean_diff": mean_diff,
      "ci_level": ci_level,
      "ci_lower": ci_lower,
      "ci_upper": ci_upper,
      "n_bootstraps": n_bootstraps,
  }


def holm_bonferroni_correction(
    p_values: Dict[str, float], alpha: float = ALPHA
) -> Dict[str, Dict[str, Any]]:
  """Applies step-down Holm-Bonferroni correction for multiple hypothesis testing."""
  sorted_items = sorted(p_values.items(), key=lambda x: x[1])
  m = len(sorted_items)
  results = {}

  for rank, (name, p_val) in enumerate(sorted_items, start=1):
    threshold = alpha / (m - rank + 1)
    results[name] = {
        "raw_p": p_val,
        "holm_rank": rank,
        "adjusted_threshold": threshold,
        "rejected": p_val <= threshold,
    }

  return results


def summarize_continuous_metric(
    values_baseline: List[float], values_proposed: List[float]
) -> Dict[str, Any]:
  """Computes descriptive and paired non-parametric tests (Wilcoxon) for continuous metrics."""
  b = np.array(values_baseline)
  p = np.array(values_proposed)

  diff = p - b
  try:
    if np.all(diff == 0):
      stat, p_val = 0.0, 1.0
    else:
      w_res = stats.wilcoxon(b, p)
      stat, p_val = float(w_res.statistic), float(w_res.pvalue)
  except Exception:
    stat, p_val = 0.0, 1.0

  return {
      "baseline": {
          "mean": float(np.mean(b)),
          "std": float(np.std(b, ddof=1)) if len(b) > 1 else 0.0,
          "median": float(np.median(b)),
          "iqr": float(stats.iqr(b)),
      },
      "proposed": {
          "mean": float(np.mean(p)),
          "std": float(np.std(p, ddof=1)) if len(p) > 1 else 0.0,
          "median": float(np.median(p)),
          "iqr": float(stats.iqr(p)),
      },
      "difference_mean": float(np.mean(diff)),
      "wilcoxon_stat": stat,
      "wilcoxon_p_value": p_val,
  }
