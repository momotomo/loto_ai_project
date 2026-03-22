# Cross-Loto Architecture Comparison Report

Generated: 2026-03-22T16:51:26.132495+00:00

## Execution Conditions

- **loto_types**: loto6, loto7, miniloto
- **preset**: default
- **seeds**: [42, 123, 456, 789, 999]
- **alpha**: 0.05
- **evaluation_model_variants**: legacy, multihot, deepsets, settransformer
- **run_id**: 20260322T165126Z_2026-03-22_archcomp_full_b
- **source_summary_paths**:
  - `loto6`: `/Users/kasuyatomohiro/loto_ai_project/campaigns/2026-03-22_archcomp_full_b/comparison_summary_loto6.json`
  - `loto7`: `/Users/kasuyatomohiro/loto_ai_project/campaigns/2026-03-22_archcomp_full_b/comparison_summary_loto7.json`
  - `miniloto`: `/Users/kasuyatomohiro/loto_ai_project/campaigns/2026-03-22_archcomp_full_b/comparison_summary_miniloto.json`

**Runs per loto_type:**

| loto_type | run_count |
|-----------|-----------|
| loto6 | 5 |
| loto7 | 5 |
| miniloto | 5 |

## Variant Metrics Summary

Metrics are averaged across all loto_types evaluated. Lower is better for logloss/brier/ece.

| rank | variant | logloss (mean±std) | brier (mean±std) | ece (mean±std) | loto_types |
|------|---------|-------------------|-----------------|---------------|------------|
| 1 | **settransformer** | 0.5371±0.0793 | 0.1759±0.0363 | 0.1840±0.0621 | loto6, loto7, miniloto |
| 2 | **multihot** | 0.5471±0.0709 | 0.1801±0.0325 | 0.1979±0.0517 | loto6, loto7, miniloto |
| 3 | **deepsets** | 0.5561±0.0721 | 0.1840±0.0337 | 0.2105±0.0483 | loto6, loto7, miniloto |
| 4 | **legacy** | 0.5603±0.0708 | 0.1860±0.0331 | 0.2146±0.0477 | loto6, loto7, miniloto |

## Promote / Hold Summary

| variant | promote_count | hold_count | promote_rate | consistent_promote | consistent_hold |
|---------|--------------|-----------|-------------|-------------------|----------------|
| **settransformer** | 0 | 15 | 0.0% | ✗ | ✓ |
| **multihot** | 0 | 15 | 0.0% | ✗ | ✓ |
| **deepsets** | 0 | 15 | 0.0% | ✗ | ✓ |
| **legacy** | 0 | 0 | 0.0% | ✗ | ✗ |

- **deepsets** held in: loto6, loto7, miniloto
- **multihot** held in: loto6, loto7, miniloto
- **settransformer** held in: loto6, loto7, miniloto

## Pairwise Comparison Summary

**ci_wins**: runs where bootstrap CI upper bound < 0 (challenger better than baseline).  
**perm_wins**: runs where permutation test p-value < alpha.  
**both_pass**: runs where both CI and permutation test pass simultaneously.

| comparison | ci_wins / runs | perm_wins / runs | both_pass / runs | both_pass rate |
|------------|---------------|-----------------|-----------------|---------------|
| settransformer_vs_deepsets | 3/15 | 4/15 | 3/15 | 20.0% |
| settransformer_vs_multihot | 1/15 | 2/15 | 1/15 | 6.7% |
| deepsets_vs_multihot | 2/15 | 4/15 | 1/15 | 6.7% |
| settransformer_vs_legacy | 1/15 | 2/15 | 1/15 | 6.7% |
| deepsets_vs_legacy | 0/15 | 2/15 | 0/15 | 0.0% |
| multihot_vs_legacy | 2/15 | 3/15 | 2/15 | 13.3% |
| deepsets_vs_best_static | 0/15 | 3/15 | 0/15 | 0.0% |
| legacy_vs_best_static | 0/15 | 3/15 | 0/15 | 0.0% |
| multihot_vs_best_static | 0/15 | 3/15 | 0/15 | 0.0% |
| settransformer_vs_best_static | 0/15 | 2/15 | 0/15 | 0.0% |

⭐ = both_pass_count / run_count ≥ 0.5 (meaningful signal)

### Per-Loto Pairwise Breakdown

**settransformer_vs_deepsets**

| loto_type | ci_wins / runs | perm_wins / runs | both_pass / runs |
|-----------|---------------|-----------------|-----------------|
| loto6 | 0/5 | 1/5 | 0/5 |
| loto7 | 0/5 | 0/5 | 0/5 |
| miniloto | 3/5 | 3/5 | 3/5 |

**settransformer_vs_multihot**

| loto_type | ci_wins / runs | perm_wins / runs | both_pass / runs |
|-----------|---------------|-----------------|-----------------|
| loto6 | 1/5 | 2/5 | 1/5 |
| loto7 | 0/5 | 0/5 | 0/5 |
| miniloto | 0/5 | 0/5 | 0/5 |

**deepsets_vs_multihot**

| loto_type | ci_wins / runs | perm_wins / runs | both_pass / runs |
|-----------|---------------|-----------------|-----------------|
| loto6 | 1/5 | 1/5 | 1/5 |
| loto7 | 0/5 | 0/5 | 0/5 |
| miniloto | 1/5 | 3/5 | 0/5 |

**settransformer_vs_legacy**

| loto_type | ci_wins / runs | perm_wins / runs | both_pass / runs |
|-----------|---------------|-----------------|-----------------|
| loto6 | 0/5 | 1/5 | 0/5 |
| loto7 | 0/5 | 0/5 | 0/5 |
| miniloto | 1/5 | 1/5 | 1/5 |

**deepsets_vs_legacy**

| loto_type | ci_wins / runs | perm_wins / runs | both_pass / runs |
|-----------|---------------|-----------------|-----------------|
| loto6 | 0/5 | 0/5 | 0/5 |
| loto7 | 0/5 | 0/5 | 0/5 |
| miniloto | 0/5 | 2/5 | 0/5 |

**multihot_vs_legacy**

| loto_type | ci_wins / runs | perm_wins / runs | both_pass / runs |
|-----------|---------------|-----------------|-----------------|
| loto6 | 0/5 | 1/5 | 0/5 |
| loto7 | 1/5 | 1/5 | 1/5 |
| miniloto | 1/5 | 1/5 | 1/5 |

**deepsets_vs_best_static**

| loto_type | ci_wins / runs | perm_wins / runs | both_pass / runs |
|-----------|---------------|-----------------|-----------------|
| loto6 | 0/5 | 0/5 | 0/5 |
| loto7 | 0/5 | 0/5 | 0/5 |
| miniloto | 0/5 | 3/5 | 0/5 |

**legacy_vs_best_static**

| loto_type | ci_wins / runs | perm_wins / runs | both_pass / runs |
|-----------|---------------|-----------------|-----------------|
| loto6 | 0/5 | 0/5 | 0/5 |
| loto7 | 0/5 | 1/5 | 0/5 |
| miniloto | 0/5 | 2/5 | 0/5 |

**multihot_vs_best_static**

| loto_type | ci_wins / runs | perm_wins / runs | both_pass / runs |
|-----------|---------------|-----------------|-----------------|
| loto6 | 0/5 | 1/5 | 0/5 |
| loto7 | 0/5 | 0/5 | 0/5 |
| miniloto | 0/5 | 2/5 | 0/5 |

**settransformer_vs_best_static**

| loto_type | ci_wins / runs | perm_wins / runs | both_pass / runs |
|-----------|---------------|-----------------|-----------------|
| loto6 | 0/5 | 1/5 | 0/5 |
| loto7 | 0/5 | 1/5 | 0/5 |
| miniloto | 0/5 | 0/5 | 0/5 |

## Recommendation

**Recommended next action**: `hold`
**Recommended challenger**: settransformer
**Keep production as-is**: Yes
**Whether to try PMA / ISAB next**: No ✗

### Evidence Summary

- Best variant by logloss: **settransformer**
- Consistently promoting variants: []
- Clear pairwise winner: None
- Total runs across loto_types: 15

### Blockers to Promotion

- No variant consistently passed promotion guardrails across loto_types.

### Next Experiment Recommendations

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
| **deepsets** | 0 | 3 | 12 |
| **legacy** | 0 | 1 | 14 |
| **multihot** | 0 | 1 | 14 |
| **settransformer** | 0 | 1 | 14 |

---
*Report generated by cross_loto_report.py v1*
