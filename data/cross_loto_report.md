# Cross-Loto Architecture Comparison Report

Generated: 2026-03-21T05:42:39.231470+00:00

## Execution Conditions

- **loto_types**: loto6
- **preset**: archcomp
- **seeds**: [42, 123, 456]
- **alpha**: 0.05
- **evaluation_model_variants**: legacy, multihot, deepsets, settransformer
- **run_id**: 20260321T054239Z_cross_loto
- **source_summary_paths**:
  - `loto6`: `/Users/kasuyatomohiro/loto_ai_project/data/comparison_summary_loto6.json`

**Runs per loto_type:**

| loto_type | run_count |
|-----------|-----------|
| loto6 | 1 |

## Variant Metrics Summary

Metrics are averaged across all loto_types evaluated. Lower is better for logloss/brier/ece.

| rank | variant | logloss (mean±std) | brier (mean±std) | ece (mean±std) | loto_types |
|------|---------|-------------------|-----------------|---------------|------------|
| 1 | **multihot** | 0.6931±0.0000 | 0.2500±0.0000 | 0.3597±0.0000 | loto6 |
| 2 | **deepsets** | 0.6931±0.0000 | 0.2500±0.0000 | 0.3606±0.0000 | loto6 |
| 3 | **legacy** | 0.6971±0.0000 | 0.2520±0.0000 | 0.3621±0.0000 | loto6 |
| 4 | **settransformer** | 0.6984±0.0000 | 0.2526±0.0000 | 0.3637±0.0000 | loto6 |

## Promote / Hold Summary

| variant | promote_count | hold_count | promote_rate | consistent_promote | consistent_hold |
|---------|--------------|-----------|-------------|-------------------|----------------|
| **multihot** | 0 | 1 | 0.0% | ✗ | ✓ |
| **deepsets** | 0 | 1 | 0.0% | ✗ | ✓ |
| **legacy** | 0 | 0 | 0.0% | ✗ | ✗ |
| **settransformer** | 0 | 1 | 0.0% | ✗ | ✓ |

- **deepsets** held in: loto6
- **multihot** held in: loto6
- **settransformer** held in: loto6

## Pairwise Comparison Summary

**ci_wins**: runs where bootstrap CI upper bound < 0 (challenger better than baseline).  
**perm_wins**: runs where permutation test p-value < alpha.  
**both_pass**: runs where both CI and permutation test pass simultaneously.

| comparison | ci_wins / runs | perm_wins / runs | both_pass / runs | both_pass rate |
|------------|---------------|-----------------|-----------------|---------------|
| settransformer_vs_deepsets | 0/1 | 0/1 | 0/1 | 0.0% |
| settransformer_vs_multihot | 0/1 | 0/1 | 0/1 | 0.0% |
| deepsets_vs_multihot | 0/1 | 0/1 | 0/1 | 0.0% |
| settransformer_vs_legacy | 0/1 | 0/1 | 0/1 | 0.0% |
| deepsets_vs_legacy | 1/1 | 1/1 | 1/1 | 100.0% ⭐ |
| multihot_vs_legacy | 0/1 | 0/1 | 0/1 | 0.0% |
| deepsets_vs_best_static | 0/1 | 1/1 | 0/1 | 0.0% |
| legacy_vs_best_static | 0/1 | 1/1 | 0/1 | 0.0% |
| multihot_vs_best_static | 0/1 | 1/1 | 0/1 | 0.0% |
| settransformer_vs_best_static | 0/1 | 1/1 | 0/1 | 0.0% |

⭐ = both_pass_count / run_count ≥ 0.5 (meaningful signal)

### Per-Loto Pairwise Breakdown

**settransformer_vs_deepsets**

| loto_type | ci_wins / runs | perm_wins / runs | both_pass / runs |
|-----------|---------------|-----------------|-----------------|
| loto6 | 0/1 | 0/1 | 0/1 |

**settransformer_vs_multihot**

| loto_type | ci_wins / runs | perm_wins / runs | both_pass / runs |
|-----------|---------------|-----------------|-----------------|
| loto6 | 0/1 | 0/1 | 0/1 |

**deepsets_vs_multihot**

| loto_type | ci_wins / runs | perm_wins / runs | both_pass / runs |
|-----------|---------------|-----------------|-----------------|
| loto6 | 0/1 | 0/1 | 0/1 |

**settransformer_vs_legacy**

| loto_type | ci_wins / runs | perm_wins / runs | both_pass / runs |
|-----------|---------------|-----------------|-----------------|
| loto6 | 0/1 | 0/1 | 0/1 |

**deepsets_vs_legacy**

| loto_type | ci_wins / runs | perm_wins / runs | both_pass / runs |
|-----------|---------------|-----------------|-----------------|
| loto6 | 1/1 | 1/1 | 1/1 |

**multihot_vs_legacy**

| loto_type | ci_wins / runs | perm_wins / runs | both_pass / runs |
|-----------|---------------|-----------------|-----------------|
| loto6 | 0/1 | 0/1 | 0/1 |

**deepsets_vs_best_static**

| loto_type | ci_wins / runs | perm_wins / runs | both_pass / runs |
|-----------|---------------|-----------------|-----------------|
| loto6 | 0/1 | 1/1 | 0/1 |

**legacy_vs_best_static**

| loto_type | ci_wins / runs | perm_wins / runs | both_pass / runs |
|-----------|---------------|-----------------|-----------------|
| loto6 | 0/1 | 1/1 | 0/1 |

**multihot_vs_best_static**

| loto_type | ci_wins / runs | perm_wins / runs | both_pass / runs |
|-----------|---------------|-----------------|-----------------|
| loto6 | 0/1 | 1/1 | 0/1 |

**settransformer_vs_best_static**

| loto_type | ci_wins / runs | perm_wins / runs | both_pass / runs |
|-----------|---------------|-----------------|-----------------|
| loto6 | 0/1 | 1/1 | 0/1 |

## Recommendation

**Recommended next action**: `run_more_seeds`
**Recommended challenger**: multihot
**Keep production as-is**: Yes
**Whether to try PMA / ISAB next**: No ✗

### Evidence Summary

- Best variant by logloss: **multihot**
- Consistently promoting variants: []
- Clear pairwise winner: None
- Total runs across loto_types: 1

### Blockers to Promotion

- No variant consistently passed promotion guardrails across loto_types.
- settransformer_vs_deepsets: no run showed both CI and permutation significance.

### Next Experiment Recommendations

- Increase seed count to ≥5 and re-run cross-loto comparison.
- settransformer vs deepsets advantage is not yet clear. PMA / ISAB exploration should wait until settransformer shows consistent benefit.

## Production Change Rationale

**Production model is NOT changed** by this comparison run.  
Final training (`--no_skip_final_train`) was either skipped (default) or no variant met the consistent-promotion threshold across loto_types.

To change production, you would need:
1. A `consider_promotion` verdict from this report
2. Manual confirmation of the candidate variant
3. Re-run with `--no_skip_final_train` for the chosen `--model_variant`

## Decision Rules

| Condition | Next Action |
|-----------|-------------|
| At least one non-legacy variant promoted in ≥50% of loto_types | `consider_promotion` |
| No consistent promotion, but some pairwise pair has both_pass_count/run_count ≥ 0.5 | `run_more_seeds` |
| Neither condition above is met | `hold` |

**Promotion to production requires:**
- `consider_promotion` verdict from cross-loto summary
- Manual review of per-loto details and calibration recommendation

**PMA / ISAB exploration requires:**
- `settransformer_vs_deepsets` overall `both_pass_count / run_count ≥ 0.5`
  (confirms attention benefit before investing in deeper pooling architectures)

**Thresholds:**
- `CONSISTENT_PROMOTE_THRESHOLD = 0.5` — variant promoted in ≥ ceil(N × 0.5) loto_types
- `PAIRWISE_SIGNAL_THRESHOLD = 0.5` — both_pass_count / run_count for PMA/ISAB signal

## Calibration Recommendations

| variant | none | temperature | isotonic |
|---------|------|-------------|---------|
| **deepsets** | 1 | 0 | 0 |
| **legacy** | 0 | 1 | 0 |
| **multihot** | 1 | 0 | 0 |
| **settransformer** | 0 | 1 | 0 |

---
*Report generated by cross_loto_report.py v1*
