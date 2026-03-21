# EVALUATION

## walk-forward (rolling-origin)
- 評価対象は draw 単位の時系列サンプルで、各サンプルは直近 `LOOKBACK_WINDOW` 回を入力、次回 draw の multi-hot を正解とする。
- 入力表現は 4 系統ある。`legacy` は既存の tabular 特徴、`multihot` は 1 draw を `max_num` 次元 multi-hot とし各番号に `hit / frequency / gap` を付与した時系列、`deepsets` は 1 draw を当選番号の集合として扱い各 element に `number_norm / frequency / gap` を付与して shared encoder -> mean pooling -> lookback LSTM に渡す。`settransformer` は `deepsets` と同一 element feature を用いつつ、pooling 前に SetAttentionBlock (SAB) で集合内要素の相互作用を付加した上で lookback LSTM に渡す。
- `legacy_holdout` は従来互換の単発 split。
- `walk_forward` は expanding-window 方式で train を広げ、固定長 test window を複数 fold で評価する。
- `legacy_holdout` と `walk_forward` はどちらも leak-free 評価で、scaler は train 期間だけで fit する。
- calibration は各 fold の train 末尾から切り出した calibration split だけで fit し、test には apply のみを行う。
- 一方で final model は運用モデル生成用なので、評価後に全データで fit し直して保存する。評価値と運用モデル生成は分離して考える。
- `smoke` preset は plumbing 検証用なので 0-epoch で動かし、配線・artifact・CLI/UI 互換の確認に使う。

## 指標
- LogLoss (BCE): draw × number の二値確率損失。
- Brier score: 確率予測の二乗誤差。
- Calibration: 予測確率を 10 bin に分けた平均予測確率と実測率。
- ECE: reliability bins から計算する expected calibration error。
- Top-k overlap: `k = pick_count` として、予測上位 k 個と実当選集合の重なり数ヒストグラム。

## post-hoc calibration
- `saved_calibration_method` は production 保存時に使う method を指定する。既定は `none`。
- `evaluation_calibration_methods` は評価時に比較する method 群で、既定は `none,temperature,isotonic`。
- `temperature` は依存が軽く、グローバルな確率スケール補正として扱いやすい。
- `isotonic` は単調変換で柔軟だが過学習しやすいため、calibration split を leak-free に切り出して比較専用に使う。
- fold ごとに raw と post-calibration の logloss / Brier / ECE を保存し、variant ごとに guardrail 付きで最終候補 method を選ぶ。

## 統計的比較
- `statistical_tests` では draw 単位の logloss 差 `candidate - reference` を保存する。負の値ほど candidate が良い。
- 各比較には bootstrap CI と paired permutation test を付与する。
- 比較時の予測列は各 variant の `selected_calibration_method` に揃える。raw 比較は `calibration_selection.raw_metrics` 側で残す。
- 最低限の比較対象:
  - `legacy_vs_best_static`
  - `multihot_vs_best_static`
  - `multihot_vs_legacy`
  - `deepsets_vs_best_static`
  - `deepsets_vs_legacy`
  - `deepsets_vs_multihot`
  - `settransformer_vs_best_static`
  - `settransformer_vs_legacy`
  - `settransformer_vs_multihot`（補助）
  - `settransformer_vs_deepsets`（補助）
- 主比較は proper scoring rule である logloss を使う。Top-k は補助指標として解釈する。

## 採用判定ルール
- `decision_summary.rule` にルール文字列を保存する。
- 現行ルールは challenger variant ごとに `variant_vs_best_static` と `variant_vs_legacy` の両方で、`bootstrap_ci.upper < 0` かつ `permutation_test.p_value < alpha` を満たし、さらに候補 calibration 後の logloss / Brier が raw 比で悪化せず、ECE が raw 改善または threshold 内にあるときだけ昇格候補とする。
- 条件を満たさない場合は既存 production variant を維持する。

## 集計評価と回別照合の違い
- `eval_report_*.json` は fold 単位と aggregate の集計指標を見るための artifact。
- `prediction_history_*.json` は各 draw の `predicted_top_k` と `actual_numbers` を照合し、「どの回で何個当たったか」を確認するための artifact。
- 集計で平均が良く見えても、回別に見ると hit が偏っていることがある。両方を合わせて解釈する。
- Streamlit では評価タブが集計、`✅ 実績との照合` タブが回別の比較を担当する。

## static baseline と online baseline
- `static_baselines`
  - `uniform`: 一様確率。
  - `frequency`: train 期間の出現頻度を固定で使う。
  - `gap`: train 末尾時点の gap 状態から作る固定予測。
- `online_baselines`
  - `frequency_online`: test 実測を逐次取り込んで頻度更新。
  - `gap_online`: test 実測で last-seen を逐次更新。
- モデルは test 中に再学習しないため、主比較は `static_baselines` に置く。

## リーク防止
- scaler は fold ごとに train 期間の観測済みデータだけで fit する。
- test 期間は transform のみを使う。
- `frequency` / `gap` など番号レベル特徴は prefix-causal に構築し、将来の draw を追加しても過去行の値が変わらないことを前提にする。
- calibration / overlap / baseline も fold ごとに算出し、最後に集計する。
- calibration fit に使う split は fold の train 末尾だけで、test draw の情報は使わない。
- prediction history も同じ test 区間の予測からだけ組み立て、future draw の情報は使わない。
- online baseline は test 実測を使うが、予測後にのみ状態更新する。

## preset の使い分け

| preset | 目的 | eval_epochs | walk_forward_folds | 備考 |
|--------|------|-------------|-------------------|------|
| `smoke` | 配線・artifact 確認 | 0 | 1 | 0-epoch なので logloss 差は無意味 |
| `fast` | 素早い動作確認 | 4 | 3 | CI / 手元 sanity check 向け |
| `default` | 通常運用 | 8 | 5 | Kaggle 無料枠で実行可能 |
| `archcomp` | deepsets vs settransformer 比較専用 | 6 | 3 | 多 seed 比較と summary 生成に使う |

### コマンド例

- 通常: `venv/bin/python train_prob_model.py --loto_type loto6 --model_variant legacy --evaluation_model_variants legacy,multihot,deepsets,settransformer`
- calibration 比較つき: `venv/bin/python train_prob_model.py --loto_type loto6 --model_variant legacy --evaluation_model_variants legacy,multihot,deepsets,settransformer --saved_calibration_method none --evaluation_calibration_methods none,temperature,isotonic`
- 短時間確認: `venv/bin/python train_prob_model.py --loto_type loto6 --preset fast`
- 最小 smoke (全 4 variant): `venv/bin/python train_prob_model.py --loto_type loto6 --preset smoke --model_variant legacy --evaluation_model_variants legacy,multihot,deepsets,settransformer`
- settransformer vs deepsets 集中比較 (smoke): `venv/bin/python train_prob_model.py --loto_type loto6 --preset smoke --model_variant legacy --evaluation_model_variants legacy,deepsets,settransformer --skip_final_train`
- deepsets 保存 smoke: `venv/bin/python train_prob_model.py --loto_type loto6 --preset smoke --model_variant deepsets --evaluation_model_variants legacy,multihot,deepsets,settransformer --skip_final_train`
- settransformer 保存 smoke: `venv/bin/python train_prob_model.py --loto_type loto6 --preset smoke --model_variant settransformer --evaluation_model_variants legacy,deepsets,settransformer --skip_final_train`
- 評価だけ更新したい場合: `venv/bin/python train_prob_model.py --loto_type loto6 --preset smoke --skip_final_train`
- 実験追跡込み: `venv/bin/python scripts/run_experiment.py --config-json '{"loto_type":"loto6","preset":"smoke","seed":42,"model_variant":"deepsets","evaluation_model_variants":"legacy,multihot,deepsets,settransformer","refresh_data":false,"skip_final_train":true}'`

## architecture comparison preset (archcomp) と多 seed 比較

### archcomp preset の役割
- smoke preset は 0-epoch のため、deepsets と settransformer の architecture 差が現れない。
- `archcomp` preset (eval_epochs=6, walk_forward_folds=3, patience=3) はある程度学習させた上で両者を比較するための専用設定。
- production 保存を変えない運用（`--skip_final_train`）と組み合わせて使う。

### 多 seed 比較の実行

```bash
# 3 seeds で deepsets と settransformer を比較し summary を生成する
venv/bin/python scripts/run_multi_seed.py \
    --loto_type loto6 \
    --preset archcomp \
    --seeds 42,123,456 \
    --model_variant legacy \
    --evaluation_model_variants legacy,multihot,deepsets,settransformer \
    --saved_calibration_method none \
    --evaluation_calibration_methods none,temperature,isotonic \
    --run_root runs
```

各 seed の結果は `runs/` 以下に個別ディレクトリとして保存され、
集計サマリが `data/comparison_summary_loto6.json` に書き出される。

### comparison summary artifact の読み方

`data/comparison_summary_{loto_type}.json` に以下が含まれる:

```json
{
  "schema_version": 1,
  "loto_type": "loto6",
  "preset": "archcomp",
  "seeds": [42, 123, 456],
  "run_count": 3,
  "variants": {
    "deepsets": {
      "run_count": 3,
      "logloss": { "mean": 0.123, "std": 0.005, "values": [...] },
      "brier":   { "mean": 0.045, "std": 0.002, "values": [...] },
      "ece":     { "mean": 0.012, "std": 0.003, "values": [...] },
      "calibration_recommendations": { "none": 2, "temperature": 1 },
      "promote_count": 0,
      "hold_count": 3
    },
    ...
  },
  "pairwise_comparisons": {
    "settransformer_vs_deepsets": {
      "run_count": 3,
      "ci_wins": 1,           // runs where bootstrap_ci.upper < 0
      "permutation_wins": 2,  // runs where permutation_test.p_value < alpha
      "both_pass_count": 0    // runs where both pass simultaneously
    },
    ...
  }
}
```

### 判断の読み方

1. `variants.<name>.logloss.mean / std` で各 variant の性能と安定性を確認する。
2. `pairwise_comparisons.settransformer_vs_deepsets` を見る:
   - `both_pass_count >= 2/3` → settransformer が consistently better な証拠
   - `ci_wins = 0` かつ `permutation_wins = 0` → smoke と同様、hold が妥当
   - `ci_wins + permutation_wins` のどちらかだけ多い → 傾向あり、追加 seed や default preset での確認を推奨
3. `variants.<name>.promote_count` が各 seed でどれだけ promotion guardrail を通過したかを示す。
4. 「smoke では hold でも archcomp ではどうか」を比較することが、ISAB や PMA 等の次の variant 追加判断の基礎となる。

## settransformer vs deepsets 比較の読み方
- smoke では 0-epoch のため logloss 差は無意味。統計的優位の判定も hold になるのが通常。
- より安定した比較には `archcomp` または `fast` / `default` preset を使い `evaluation_model_variants legacy,deepsets,settransformer` を指定し、`settransformer_vs_deepsets` の `bootstrap_ci.upper` と `permutation_test.p_value` を確認する。
- 多 seed で確認する場合は `scripts/run_multi_seed.py` を使い、`comparison_summary` の `pairwise_comparisons.settransformer_vs_deepsets` を読む。
- `input_summary.pooling` が `mean` (deepsets) vs `mean_after_attention` (settransformer) で、それ以外の構造は同一。純粋な attention 有無の効果が分離できる設計。
- 次の候補 (PMA / ISAB) を追加する前に、まずこの比較 summary を確認し deepsets と settransformer のどちらが基準として優れているかを把握することを推奨する。

## cross-loto comparison と decision artifact

### 目的

単一の loto_type だけの比較では、結果がそのゲームの特性（番号数・pick 数・データ量）に偏る可能性がある。
cross-loto comparison は miniloto / loto6 / loto7 を横断して同じ variant を比較することで、
「どの variant がどのゲームでも安定して有望か」を判断しやすくする。

### 実行コマンド

```bash
# 全 loto_type を 3 seeds で比較（archcomp preset）
venv/bin/python scripts/run_cross_loto.py \
    --loto_types loto6,loto7,miniloto \
    --preset archcomp \
    --seeds 42,123,456 \
    --evaluation_model_variants legacy,multihot,deepsets,settransformer \
    --run_root runs

# 既存の comparison_summary を再集計するだけ（学習を省略）
venv/bin/python scripts/run_cross_loto.py \
    --loto_types loto6,loto7,miniloto \
    --skip_training
```

### 出力 artifact

| artifact | 場所 | 内容 |
|----------|------|------|
| `comparison_summary_{loto_type}.json` | `data/` | 各 loto_type の per-seed 集計 |
| `cross_loto_summary.json` | `data/` | loto_type 横断の variant 比較・ランキング |
| `recommendation.json` | `data/` | 次に取るべき行動の機械可読・人間可読な推奨 |

### cross_loto_summary.json の構造

```json
{
  "schema_version": 1,
  "loto_types": ["loto6", "loto7", "miniloto"],
  "preset": "archcomp",
  "seeds": [42, 123, 456],
  "overall_summary": {
    "variants": {
      "deepsets": {
        "loto_types_evaluated": ["loto6", "loto7", "miniloto"],
        "logloss": {"mean": 0.123, "std": 0.005, "per_loto": {...}},
        "brier":   {"mean": 0.045, "std": 0.002, "per_loto": {...}},
        "ece":     {"mean": 0.012, "std": 0.003, "per_loto": {...}},
        "promote_count_total": 0,
        "hold_count_total": 9
      }
    }
  },
  "variant_ranking": {
    "by_logloss": [{"rank": 1, "variant": "legacy", "mean": 0.290}, ...],
    "by_brier":   [...],
    "by_ece":     [...],
    "promote_counts": {"deepsets": {"promote_count": 0, "hold_count": 9, "promote_rate": 0.0}},
    "calibration_recommendations": {"deepsets": {"none": 9}}
  },
  "pairwise_comparison_summary": {
    "settransformer_vs_deepsets": {
      "per_loto": {"loto6": {"run_count": 3, "ci_wins": 1, ...}},
      "overall":  {"run_count": 9, "ci_wins": 2, "permutation_wins": 3, "both_pass_count": 1}
    }
  },
  "promotion_recommendation_summary": {
    "deepsets": {
      "promoted_in": [],
      "held_in": ["loto6", "loto7", "miniloto"],
      "consistent_promote": false,
      "consistent_hold": true
    }
  }
}
```

### recommendation.json の構造

```json
{
  "schema_version": 1,
  "based_on": "cross_loto_summary",
  "recommended_next_action": "hold",
  "recommended_challenger": "deepsets",
  "keep_production_as_is": true,
  "evidence_summary": {
    "best_variant_by_logloss": "legacy",
    "consistent_promote_variants": [],
    "pairwise_clear_winner": null
  },
  "blockers_to_promotion": [
    "No variant consistently passed promotion guardrails across loto_types."
  ],
  "whether_to_try_pma_or_isab_next": false,
  "next_experiment_recommendations": [...]
}
```

### cross-loto summary の読み方

**まず `data/cross_loto_report.md` を開く** — 全情報を人が読みやすい Markdown にまとめたレポート。
詳細を掘り下げたいときに `data/cross_loto_summary.json` / `recommendation.json` を参照する。
表計算で確認したいときは `data/variant_metrics.csv` / `data/pairwise_comparisons.csv` を使う。

1. `variant_ranking.by_logloss` で全体傾向を確認する（lower = better）。
2. `pairwise_comparison_summary.settransformer_vs_deepsets.overall` を見る:
   - `both_pass_count / run_count >= 0.5` → attention (SAB) に明確なメリットあり
   - それ以下 → まだ証拠不十分、追加 seed や default preset での確認を推奨
3. `promotion_recommendation_summary.<variant>.consistent_promote` が true なら、
   その variant は過半数の loto_type で promotion guardrail を通過している。
4. `recommendation.json` の `recommended_next_action` を基本方針とする:
   - `hold` → production を変えず追加実験を検討
   - `run_more_seeds` → seed 数を増やして信頼区間を絞る
   - `consider_promotion` → 候補 variant の本番学習を検討

### 出力 artifact（全形式）

| artifact | 場所 | 内容 | 用途 |
|----------|------|------|------|
| `comparison_summary_{loto_type}.json` | `data/` | 各 loto_type の per-seed 集計 | raw JSON |
| `cross_loto_summary.json` | `data/` | loto_type 横断の variant 比較・ランキング | raw JSON |
| `recommendation.json` | `data/` | 次に取るべき行動の推奨 | raw JSON |
| `cross_loto_report.md` | `data/` | 全情報を人が読める Markdown evidence pack | まず読むもの |
| `variant_metrics.csv` | `data/` | variant ごとの logloss/brier/ece + promote/hold | 表計算 |
| `pairwise_comparisons.csv` | `data/` | pairwise 比較の per-loto + overall | 表計算 |
| `recommendation_summary.csv` | `data/` | recommendation 1 行サマリ | 表計算 |

### レポートだけ再生成する（学習なし）

```bash
# 既存の cross_loto_summary.json / recommendation.json から Markdown + CSV を再生成
venv/bin/python scripts/run_cross_loto.py --report_only
```

### 判断ルールの明文化（decision rules）

| 条件 | next_action |
|------|------------|
| 非 legacy variant が ≥50% の loto_type で昇格判定を通過 | `consider_promotion` |
| 上記なし、かつ pairwise で both_pass_count/run_count ≥ 0.5 の pair が存在 | `run_more_seeds` |
| 上記いずれも該当しない | `hold` |

しきい値:
- `CONSISTENT_PROMOTE_THRESHOLD = 0.5` — variant が昇格した loto_type 数 / 評価 loto_type 数 ≥ 0.5
- `PAIRWISE_SIGNAL_THRESHOLD = 0.5` — pairwise の both_pass_count / run_count ≥ 0.5

これらのしきい値は `cross_loto_summary.py` と `cross_loto_report.py` のコードと同期している。

### production を変えない前提での読み方

- `keep_production_as_is: true` → production artifact は変更しない
- `skip_final_train` が既定 `True` のため、比較実行で production が上書きされることはない
- production を変える場合は明示的に `--no_skip_final_train` と `--model_variant <variant>` を指定する

### 次に PMA / ISAB / HPO に進む判断条件

- `recommendation.whether_to_try_pma_or_isab_next == true`
  （= settransformer_vs_deepsets の overall both_pass_count / run_count ≥ 0.5）
- かつ `recommended_next_action` が `consider_promotion` または `run_more_seeds`

この条件が満たされるまでは、新しい variant を追加するより cross-loto comparison の
信頼性を高めることを優先すること。`cross_loto_report.md` の「Decision Rules」節で
判断条件を確認できる。

## comparison campaign の運用

### campaign とは

単発の比較実行ではなく、名前付き・ディレクトリ保存・履歴追記型の比較実行単位。
`scripts/run_campaign.py` が entry point で、`campaign_profiles.py` で定義されたプロファイルを使って
cross-loto 比較を実行し、結果を `campaigns/<campaign_name>/` へ保存して `data/campaign_history.json` を更新する。

**「まず campaign history を見る」運用を優先する。**

### campaign profile の違い

| profile | preset | seeds | loto_types | 用途 |
|---------|--------|-------|-----------|------|
| `archcomp_lite` | fast | 2 | loto6 のみ | 軽量確認・sanity check（決定に使わない） |
| `archcomp` | archcomp | 3 | 全 3 種 | 標準 campaign（既定）。決定判断に使う |
| `archcomp_full` | default | 5 | 全 3 種 | `run_more_seeds` が ≥3 回続く場合の拡充 |

各プロファイルは `epochs / patience / batch_size` (preset 経由) + `seeds / loto_types / evaluation_model_variants / calibration_methods` を一括管理する。

### campaign の実行

```bash
# プロファイル一覧
python scripts/run_campaign.py --list_profiles

# 標準 archcomp キャンペーン
python scripts/run_campaign.py --campaign_name 2026-03-21_archcomp --profile archcomp

# 軽量確認
python scripts/run_campaign.py --campaign_name 2026-03-21_lite --profile archcomp_lite

# 拡充版（run_more_seeds が続く場合）
python scripts/run_campaign.py --campaign_name 2026-03-21_full --profile archcomp_full

# 既存の comparison_summary を再集計（学習省略）
python scripts/run_campaign.py --campaign_name 2026-03-21_archcomp --profile archcomp --skip_training
```

### campaign の artifact 一覧

| artifact | 場所 | 内容 |
|----------|------|------|
| `cross_loto_report.md` | `campaigns/<name>/` | evidence pack（campaign ごと、まず読む） |
| `campaign_metadata.json` | `campaigns/<name>/` | profile / seeds / timing |
| `cross_loto_summary.json` | `campaigns/<name>/` | variant ranking / pairwise |
| `recommendation.json` | `campaigns/<name>/` | next_action 推奨 |
| `comparison_summary_{loto_type}.json` | `campaigns/<name>/` | per-loto 集計 |
| `campaign_diff_report.md` | `data/` | 前回 campaign との差分（comparability セクション含む） |
| `campaign_history.json` | `data/` | 全 campaign 履歴 + recommendation stability（accepted-only 集計含む） |
| `campaign_history.csv` | `data/` | 表計算用履歴（accepted_for_decision_use 列含む） |
| `benchmark_lock.json` | `data/` | Decision Benchmark Policy 定義（昇格判断条件） |
| `benchmark_lock.md` | `data/` | benchmark_lock の人が読める版 |
| `campaign_acceptance.json` | `data/` | 最新 campaign の昇格判断採用可否 |
| `campaign_acceptance.md` | `data/` | campaign_acceptance の人が読める版 |
| `accepted_campaign_summary.json` | `data/` | accepted campaign のみの履歴サマリー（machine-readable） |
| `accepted_campaign_summary.md` | `data/` | accepted-only 履歴の人が読める版（raw history と区別） |
| `promotion_review_readiness.json` | `data/` | promotion review に進んでよいかの verdict（machine-readable） |
| `promotion_review_readiness.md` | `data/` | promotion review readiness の人が読める版 |
| `accepted_campaign_review_bundle.json` | `data/` | accepted campaign evidence をまとめた review bundle（machine-readable） |
| `accepted_campaign_review_bundle.md` | `data/` | **review bundle の人が読める版（昇格検討時に読む）** |

### artifact の読む順序

1. `data/governance_report.md` — 全 governance シグナル統合レポート（**最優先**）
2. `data/accepted_campaign_review_bundle.md` — accepted evidence まとめ（**昇格検討時に読む**）
3. `data/promotion_review_readiness.md` — promotion review に進んでよいか
4. `data/accepted_campaign_summary.md` — accepted-only 履歴サマリー
5. `data/campaign_acceptance.md` — この campaign は昇格判断に使えるか
6. `data/benchmark_lock.md` — 昇格判断に使える条件の定義
7. `data/campaign_diff_report.md` — 前回からの変化（何が変わったか）
8. `data/campaign_history.json` → `recommendation_stability` — 安定性トレンド
9. `campaigns/<name>/cross_loto_report.md` — 今回 campaign の evidence pack

### recommendation stability の読み方

`data/campaign_history.json` の `recommendation_stability` には以下が含まれる（新フィールドを含む）:

```json
{
  "total_campaigns": 4,
  "total_accepted_campaigns": 3,
  "latest_action": "run_more_seeds",
  "consecutive_same_action": 3,
  "consecutive_same_challenger": 4,
  "consecutive_keep_production": 4,
  "consecutive_run_more_seeds": 3,
  "consecutive_positive_signal_for_settransformer": 2,
  "consecutive_positive_signal_for_deepsets": 1,
  "consecutive_same_action_accepted_only": 2,
  "consecutive_positive_signal_for_settransformer_accepted_only": 1
}
```

全 campaign 対象の指標（`consecutive_same_action` など）はトレンド把握に使う。
昇格判断には **`_accepted_only` の指標を使うこと**。

- `consecutive_same_action_accepted_only >= 2` かつ `consider_promotion` → 昇格検討に進める根拠
- `consecutive_same_action_accepted_only >= 3` かつ `run_more_seeds` → `archcomp_full` を実行
- `consecutive_same_action >= 3` かつ `hold` → architecture 差が現れない可能性が高い
- `consecutive_positive_signal_for_settransformer_accepted_only >= 2` → PMA/ISAB 探索の動機（accepted のみ）
- `consecutive_positive_signal_for_deepsets >= 2` → deepsets が legacy に対して継続的に優位

> **注意**: `archcomp_lite` campaign は `accepted_for_decision_use=false` になるため、
> `_accepted_only` カウントには含まれない。

## governance layer の運用

### governance artifact とは

campaign 実行後に自動生成される 4 種類のシグナル集約 artifact。
`scripts/run_campaign.py` 実行時に自動生成される（`data/` 以下に保存）。

**まず `data/governance_report.md` を読む** — 他の artifact より最優先。

### governance artifact 一覧

| artifact | 形式 | 内容 |
|----------|------|------|
| `data/governance_report.md` | Markdown | 全 governance シグナルを統合した運用レポート（最優先） |
| `data/benchmark_lock.json` | JSON | Decision Benchmark Policy（昇格判断条件の定義） |
| `data/benchmark_lock.md` | Markdown | benchmark_lock の人が読める版 |
| `data/campaign_acceptance.json` | JSON | 最新 campaign の昇格判断採用可否（accepted_for_decision_use） |
| `data/campaign_acceptance.md` | Markdown | campaign_acceptance の人が読める版 |
| `data/comparability_report.json` | JSON | campaign 間の比較可能性判定結果（benchmark・loto・variant 照合） |
| `data/comparability_report.md` | Markdown | comparability_report の人が読める版 |
| `data/trend_summary.json` | JSON | 直近 N campaign の傾向（rank 推移・logloss・pairwise）+比較可能性メモ |
| `data/trend_summary.md` | Markdown | trend_summary の人が読める版 |
| `data/regression_alert.json` | JSON | 最新 campaign と過去比較の悪化シグナル+比較可能性 caution |
| `data/regression_alert.md` | Markdown | regression_alert の人が読める版 |
| `data/promotion_gate.json` | JSON | 昇格検討フェーズに進んでよいかの gate 判定（比較可能性条件含む） |
| `data/promotion_gate.md` | Markdown | promotion_gate の人が読める版 |

### Decision Benchmark Policy と accepted campaign

**comparable と accepted の違い**:

| 概念 | 定義 | 用途 |
|------|------|------|
| **comparable** | 同一条件で比較可能（benchmark 互換・loto・variant・calibration 一致） | trend 解釈・regression alert |
| **accepted** | Decision Benchmark Policy を満たす（archcomp/archcomp_full のみ・全 loto・comparability_ok） | 昇格判断の証拠として積む |

**accepted であることが必要な場面**:
- `consecutive_same_action_accepted_only` を昇格根拠にする
- promotion gate が green かどうかを昇格根拠にする
- PMA/ISAB 探索に進む判断の根拠にする

**accepted でなくてよい場面**:
- 単一 loto_type の sanity check（archcomp_lite）
- 新しい設定の動作確認

**archcomp_lite を昇格判断に使ってはいけない理由**:
- loto6 のみのため全 loto_types 条件を満たさない
- 2 seeds で variance が大きい
- benchmark_registry で archcomp/archcomp_full とは非互換

`data/campaign_acceptance.md` の Acceptance Status を読んで、`accepted_for_decision_use=true` を確認してから
promotion readiness の議論を進めること。

### benchmark registry と comparability の役割

`benchmark_registry.py` は各 campaign profile が満たすべき仕様を定義する:
- `archcomp`: 全 loto_types (3種)、3 seeds 以上、4 variant、3 calibration method
- `archcomp_full`: 全 loto_types (3種)、5 seeds 以上（archcomp と互換扱い）
- `archcomp_lite`: loto6 のみ、2 seeds（archcomp とは **非互換**）

2 campaign が比較可能（comparable）である条件:
1. **benchmark 互換** — 同一 benchmark または compatible_benchmarks に列挙されたペア
2. **loto_types 一致** — 同一の loto_type セット（hard check）
3. **variant セット一致** — evaluation_model_variants が同じ（hard check）
4. **calibration methods 一致** — evaluation_calibration_methods が同じ（hard check）
5. **seed count 十分** — benchmark minimum 以上（hard check）
6. **data fingerprint 一致** — 利用可能な場合に確認（warning のみ）

hard check 失敗 → `comparable=False`、severity=error
warning のみ → `comparable=True`、severity=warning

### comparability report の読み方

`data/comparability_report.md` の Overall Status を確認する。

| status | 意味 | 対処 |
|--------|------|------|
| ✅ COMPARABLE | 全 campaign ペアが比較可能 | trend/regression をそのまま信頼してよい |
| ⚠️ COMPARABLE WITH WARNINGS | 比較可能だが注意点あり | warning を読んで caution 付きで解釈 |
| ❌ NOT COMPARABLE | 比較不能なペアあり | trend/regression 結論を保留、条件を揃えて再実行 |

`pairs` 節に連続する campaign ペアごとの判定が入る。
`failed_checks` が hard failure、`warnings` が soft mismatches。

### trend_summary の読み方

**事前確認**: `comparability_report.md` で overall_severity を確認する。error の場合、以下のトレンド解釈は保留すること。

`data/trend_summary.md` を読む（`data/trend_summary.json` の人が読める版）。

主な確認ポイント:
- `variant_rank_history.<variant>.rank_trend` — "improving" / "worsening" / "stable" / "insufficient_data"
- `metric_trends.<variant>.logloss.trend` — logloss が増えていれば "worsening"（低いほど良い）
- `recommendation_history` — 直近 N campaign の action の流れ
- `dominant_action` — window 内で最多の recommended_next_action
- `pairwise_signal_history` — settransformer_vs_deepsets の both_pass_rate が上昇していれば attention 効果が強まっている

### regression_alert の読み方

**事前確認**: `comparability_note` フィールドと `comparability_caution` フラグを確認する。
`comparability_caution=true` の場合、alert は「条件が揃っていない比較」を反映している可能性がある。

`data/regression_alert.md` の Alert Level を確認する。

| alert_level | 意味 | 対処 |
|-------------|------|------|
| ✅ none | 有意な悪化なし | 監視継続 |
| ⚠️ low | 軽微な悪化シグナル 1 件 | 傾向確認、次 campaign で様子見 |
| 🔶 medium | 複数シグナルまたは明確な悪化 | promotion 前に調査必須 |
| 🔴 high | 重大な悪化（複数指標同時悪化 or recommendation 逆転） | promotion 禁止、原因調査 |

`suspected_causes` に自動生成されたヒューリスティック解釈が入る。

### promotion_gate の読み方

**注意**: promotion_gate には `comparability_ok` 条件が含まれる。campaign 間の比較可能性問題がある場合、gate は自動的に green にならない（comparability_error が blockers に追加される）。

`data/promotion_gate.md` で gate_status を確認する。

| gate_status | 意味 | 次のアクション |
|-------------|------|---------------|
| 🟢 green | 昇格検討フェーズに進んでよい | per-loto 詳細レビュー後に production 学習 |
| 🟡 yellow | 一部条件クリア、追加証拠推奨 | blockers を解消して再 campaign |
| 🔴 red | 条件不足または悪化あり | campaign を続けて証拠を積む |

`conditions_passed` と `blockers` に個別条件が列挙される。
gate が green でも production は自動変更されない — 手動レビュー後に明示的に学習すること。

### 何回続いたら PMA / ISAB / HPO に進むか

以下の条件が揃ったら次の variant 実装を検討できる:

1. **`accepted_for_decision_use=true` の campaign** で:
   - `recommendation.whether_to_try_pma_or_isab_next == true` が **2 回以上連続**
   - `consecutive_positive_signal_for_settransformer_accepted_only >= 2`
2. `recommended_next_action` が `consider_promotion` または `run_more_seeds`（`hold` でない）
3. `archcomp` または `archcomp_full` profile に基づく（`archcomp_lite` の campaign は不可）
4. promotion gate が green かつ manual review 完了

> **重要**: `archcomp_lite` で signal が出ても PMA/ISAB の根拠にはならない。
> `campaign_acceptance.md` で `accepted_for_decision_use=true` を確認すること。

この条件が満たされるまでは campaign を継続し、accepted campaign を積み上げることを優先する。

## 再現性メモ
- `eval_report_*.json` と `manifest_*.json` には `data_fingerprint` / `training_context` / `runtime_environment` を保存する。
- これにより data hash、preprocessing version、preset、seed、model variant、主要 hyperparameter、Python / dependency versions を artifact 単位で追える。
