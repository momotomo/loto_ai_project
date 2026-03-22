# Governance Report

Generated: 2026-03-22T02:06:53.696303+00:00

Latest campaign: **2026-03-21_archcomp_b**

> **Reading order**: Start here. For details, see `campaign_acceptance.md` → `benchmark_lock.md` → `comparability_report.md` → `trend_summary.md` → `regression_alert.md` → `promotion_gate.md` → `campaign_diff_report.md` → evidence pack in `campaigns/<name>/cross_loto_report.md`.

## Decision Benchmark Policy

> **Read this first**: Only campaigns that satisfy the Decision Benchmark Policy are counted toward promotion readiness. See `benchmark_lock.md` for the full policy.

- **Active decision benchmarks**: `archcomp`, `archcomp_full`
- **Excluded (sanity only)**: `archcomp_lite`
- **comparability_required**: `True`

## Current Campaign Acceptance

**Status**: ✅ ACCEPTED
**Counts toward promotion readiness**: ✅ `True`

Campaign `2026-03-21_archcomp_b` meets all decision benchmark policy requirements. Benchmark: `archcomp`. This campaign counts toward promotion readiness.

## Whether This Campaign Counts Toward Promotion Readiness

> ✅ **This campaign counts.** The accepted-only stability metrics below reflect this campaign.

| Accepted-Only Metric | Value |
|----------------------|-------|
| Total accepted campaigns | 2 |
| Consecutive same action (accepted only) | 2 |
| Consecutive settransformer signal (accepted only) | 0 |

## Comparability

> **Check this before trends**: trend and regression conclusions are only valid when campaigns are comparable (same benchmark, loto coverage, variants, and calibration methods).

**Status**: ⚠️ COMPARABLE

1 pair(s) checked; all comparable but 1 pair(s) have warnings.  Review warnings before drawing trend conclusions.

> ⚠️ Campaigns are comparable with caveats.  Review `comparability_report.md` before acting on trends.

## Current Recommendation

| Field | Value |
|-------|-------|
| recommended_next_action | `hold` |
| recommended_challenger | `settransformer` |
| keep_production_as_is | Yes |
| blockers_count | 1 |

> settransformer has not yet shown consistent advantage over deepsets. PMA/ISAB exploration should wait.

## Promotion Readiness Gate

**🟡 YELLOW** — Gate is YELLOW: Some conditions met (5 passed, 2 blockers). Signal is promising but not yet conclusive.

Blockers:
  - No sustained positive action signal: latest=`hold`, consecutive=2 (need ≥2 for consider_promotion or run_more_seeds)
  - No variant passed promotion guardrails (consistent_promote_variants is empty).

## Regression Alert

**✅ NONE** — No significant regression detected since the previous campaign.

## Recommendation Stability

| Metric | Value |
|--------|-------|
| Total campaigns | 2 |
| Latest action | `hold` |
| Consecutive same action | 2 |
| Consecutive same challenger | 2 |
| Consecutive keep_production | 2 |
| Consecutive run_more_seeds | 0 |
| Consecutive settransformer+ signal | 0 |
| Consecutive deepsets+ signal | 2 |


## Recent Trend Overview

Over the last 5 campaigns: dominant action = `hold`, dominant challenger = `settransformer`, keep_production streak = 2.

| variant | rank_trend | logloss_trend |
|---------|-----------|--------------|
| deepsets | stable | stable |
| legacy | stable | stable |
| multihot | stable | stable |
| settransformer | stable | stable |

## Production Status

> **Production remains unchanged.** No variant has met all promotion guardrails consistently. This is the safe default — only change after gate is GREEN and manual review.

## PMA / ISAB / HPO Guidance

settransformer has not shown consistent advantage over deepsets. PMA/ISAB/HPO exploration should wait.

Conditions to proceed to HPO (separate from PMA/ISAB):
- A single variant consistently wins across all loto_types and ≥ 5 seeds
- Gate is GREEN and promotion review confirms the advantage is genuine
- PMA/ISAB improvement (if applicable) has already been confirmed

## Accepted Campaign Review

> See `accepted_campaign_review_bundle.md` for the full accepted-only review.  
> See `accepted_campaign_summary.md` for the accepted-only history summary.  
> See `promotion_review_readiness.md` for the promotion review readiness verdict.

| Metric | Value |
|--------|-------|
| Accepted campaigns | **2** |
| Campaigns counting toward promotion readiness | 2 |
| Consecutive accepted positive signals | 0 |

Accepted campaigns: `2026-03-21_archcomp_a_retry`, `2026-03-21_archcomp_b`

## Promotion Review Readiness

**❌ NOT YET READY**

2 blocker(s) prevent entering promotion review. Accepted campaigns: 2. Run more accepted campaigns (archcomp or archcomp_full) to accumulate evidence.

Readiness blockers:
  - ❌ Latest accepted action `hold` is not a positive signal. Need ≥2 consecutive accepted campaigns with `consider_promotion` or `run_more_seeds`.
  - ❌ No variant passed promotion guardrails in the latest accepted campaign (consistent_promote_variants is empty).

> ❌ **Why current accepted evidence is not enough**: Latest accepted action `hold` is not a positive signal. Need ≥2 consecutive accepted campaigns with `consider_promotion` or `run_more_seeds`. / No variant passed promotion guardrails in the latest accepted campaign (consistent_promote_variants is empty).

---
*Governance report generated by governance_layer.py v1*
