# ARTIFACTS

## 保存場所
- `data/*_raw.csv`: 取得した生データ。
- `data/*_processed.csv`: 特徴量追加後の学習入力。
- `data/eval_report_{loto_type}.json`: 評価レポート。`legacy_holdout` と `walk_forward` を含む。
- `data/manifest_{loto_type}.json`: 生成日時、学習レンジ、最新 draw、指標要約、git commit。
- `data/prediction_history_{loto_type}.json`: 評価対象 draw ごとの予測上位番号と実当選番号の照合履歴。
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
- `prediction_history_path`
- `prediction_history_rows`
- `git_commit`

## prediction_history の要点
- `schema_version`
- `generated_at`
- `loto_type`
- `record_count`
- `records[]`
  - `draw_id`, `date`, `evaluation_mode`, `fold_index`
  - `actual_numbers`
  - `predicted_top_k`
  - `predicted_top_k_hit_count`
  - `predicted_top_k_hit_numbers`
  - `top_probability_numbers`
  - `top_probability_scores`
  - `pick_count`, `max_num`
  - `hit_rate_any`, `hit_rate_two_plus`

## 運用メモ
- `eval_report` と `manifest` は軽量なので UI / Kaggle 同期の主対象。
- `prediction_history` は集計指標の根拠を draw 単位で見返すための artifact。Streamlit の「✅ 実績との照合」タブが主な参照先。
- live 予測履歴は今回は保存しないが、将来は `pending/resolved` の 2 段階で別 artifact に拡張できるよう JSON 形式を分離している。
- モデル本体と scaler は再生成可能だが、UI 起動には必要。
- Streamlit の予測タブは `processed.csv` / `feature_cols.json` / `scaler.pkl` / `model.keras` の世代が揃っている前提。Kaggle 同期ではこれらを一時ディレクトリにまとめて取得してから最後に入れ替える。
- GitHub Actions の翌営業日実行では対象 loto_type だけ更新される。同期側も loto_type ごとに完全 bundle を判定し、完全な loto_type だけを部分更新する。
