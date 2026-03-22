# Governance Report

Generated: 2026-03-22T16:51:26.138791+00:00

Latest campaign: **2026-03-22_archcomp_full_b**

> **Reading order**: Start here. For details, see `campaign_acceptance.md` → `benchmark_lock.md` → `comparability_report.md` → `trend_summary.md` → `regression_alert.md` → `promotion_gate.md` → `campaign_diff_report.md` → evidence pack in `campaigns/<name>/cross_loto_report.md`.

## Decision Benchmark Policy

> **Read this first**: Only campaigns that satisfy the Decision Benchmark Policy are counted toward promotion readiness. See `benchmark_lock.md` for the full policy.

- **Active decision benchmarks**: `archcomp`, `archcomp_full`
- **Excluded (sanity only)**: `archcomp_lite`
- **comparability_required**: `True`

## Current Campaign Acceptance

**Status**: ✅ ACCEPTED
**Counts toward promotion readiness**: ✅ `True`

Campaign `2026-03-22_archcomp_full_b` meets all decision benchmark policy requirements. Benchmark: `archcomp_full`. This campaign counts toward promotion readiness.

## Whether This Campaign Counts Toward Promotion Readiness

> ✅ **This campaign counts.** The accepted-only stability metrics below reflect this campaign.

| Accepted-Only Metric | Value |
|----------------------|-------|
| Total accepted campaigns | 4 |
| Consecutive same action (accepted only) | 4 |
| Consecutive settransformer signal (accepted only) | 0 |

## Comparability

> **Check this before trends**: trend and regression conclusions are only valid when campaigns are comparable (same benchmark, loto coverage, variants, and calibration methods).

**Status**: ⚠️ COMPARABLE

3 pair(s) checked; all comparable but 3 pair(s) have warnings.  Review warnings before drawing trend conclusions.

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

**🔴 RED** — Gate is RED: Too many blockers (3) for promotion review. Current evidence is insufficient or contradictory.

Blockers:
  - No sustained positive action signal: latest=`hold`, consecutive=4 (need ≥2 for consider_promotion or run_more_seeds)
  - Active HIGH regression alert — resolve before promotion review.
  - No variant passed promotion guardrails (consistent_promote_variants is empty).

## Regression Alert

**🔴 HIGH** — Significant regression detected (6 signals across metrics, rankings, and/or recommendation). Investigate before any promotion decision.
Affected variants: deepsets, legacy, multihot, settransformer

## Recommendation Stability

| Metric | Value |
|--------|-------|
| Total campaigns | 4 |
| Latest action | `hold` |
| Consecutive same action | 4 |
| Consecutive same challenger | 4 |
| Consecutive keep_production | 4 |
| Consecutive run_more_seeds | 0 |
| Consecutive settransformer+ signal | 0 |
| Consecutive deepsets+ signal | 0 |

> **Signal**: `hold` for 4 consecutive campaigns. → Architecture differentiation may be limited at current scale.

## Recent Trend Overview

Over the last 5 campaigns: dominant action = `hold`, dominant challenger = `settransformer`, keep_production streak = 4.

| variant | rank_trend | logloss_trend |
|---------|-----------|--------------|
| deepsets | stable | worsening |
| legacy | stable | worsening |
| multihot | stable | worsening |
| settransformer | stable | worsening |

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
| Accepted campaigns | **4** |
| Campaigns counting toward promotion readiness | 4 |
| Consecutive accepted positive signals | 0 |

Accepted campaigns: `2026-03-21_archcomp_a_retry`, `2026-03-21_archcomp_b`, `2026-03-22_archcomp_full_a`, `2026-03-22_archcomp_full_b`

## Promotion Review Readiness

**❌ NOT YET READY**

4 blocker(s) prevent entering promotion review. Accepted campaigns: 4. Run more accepted campaigns (archcomp or archcomp_full) to accumulate evidence.

Readiness blockers:
  - ❌ Latest accepted action `hold` is not a positive signal. Need ≥2 consecutive accepted campaigns with `consider_promotion` or `run_more_seeds`.
  - ❌ Active HIGH regression alert — resolve before entering promotion review.
  - ❌ promotion_gate_red: promotion gate is RED. Resolve gate blockers before promotion review. See `promotion_gate.md` for details.
  - ❌ No variant passed promotion guardrails in the latest accepted campaign (consistent_promote_variants is empty).

> ❌ **Why current accepted evidence is not enough**: Latest accepted action `hold` is not a positive signal. Need ≥2 consecutive accepted campaigns with `consider_promotion` or `run_more_seeds`. / Active HIGH regression alert — resolve before entering promotion review. / promotion_gate_red: promotion gate is RED. Resolve gate blockers before promotion review. See `promotion_gate.md` for details. / No variant passed promotion guardrails in the latest accepted campaign (consistent_promote_variants is empty).

---
*Governance report generated by governance_layer.py v1*
