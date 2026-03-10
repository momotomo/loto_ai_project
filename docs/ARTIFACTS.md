# ARTIFACTS

## 保存場所
- `data/*_raw.csv`: 取得した生データ。
- `data/*_processed.csv`: 特徴量追加後の学習入力。
- `data/eval_report_{loto_type}.json`: 評価レポート。`legacy_holdout` と `walk_forward` を含む。
- `data/manifest_{loto_type}.json`: 生成日時、学習レンジ、最新 draw、指標要約、git commit。
- `data/*_feature_cols.json`: 学習時の特徴量列順。
- `models/*_prob.keras`: 本番推論用モデル。
- `models/*_scaler.pkl`: 本番推論用 scaler。
- `models/*_feature_cols.json`: CLI 互換のための複製。

## eval_report の要点
- 旧キー: `Model (LSTM)`, `Baselines`, `Online Baselines`
- 新キー:
  - `legacy_holdout`
  - `walk_forward.settings`
  - `walk_forward.folds`
  - `walk_forward.aggregate`

## manifest の要点
- `generated_at`
- `loto_type`
- `latest_draw_id`
- `train_range`
- `metrics_summary`
- `artifacts`
- `git_commit`

## 運用メモ
- `eval_report` と `manifest` は軽量なので UI / Kaggle 同期の主対象。
- モデル本体と scaler は再生成可能だが、UI 起動には必要。
