# ARCHITECTURE

## 全体フロー
1. `data_collector.py` が `thekyo.jp` の CSV を取得し、整合性チェック後に `data/*_raw.csv` と `data/*_processed.csv` を更新する。
2. `train_prob_model.py` が処理済み CSV から特徴量列を読み、leak-free な scaler と LSTM で `legacy_holdout` と `walk_forward` を評価する。
3. 同スクリプトが `data/eval_report_*.json`、`data/manifest_*.json`、`models/*_prob.keras`、`models/*_scaler.pkl`、`*_feature_cols.json` を生成する。
4. `app.py` が成果物を読み、確率可視化・買い目生成・評価レポート・manifest を表示する。
5. `update_system.py` は 1 → 2 → 3 → `predict.py` の smoke test を順に実行する。

## 主なファイル責務
- `config.py`: 宝くじごとの定数、lookback、サンプリング補助。
- `data_collector.py`: ダウンロード、retry、バリデーション、更新検知、特徴量付与。
- `train_prob_model.py`: 評価と本番学習、baseline 比較、artifact manifest 生成。
- `predict.py`: 保存済み成果物を使った簡易 CLI 推論。
- `app.py`: Streamlit UI、Kaggle 同期、評価レポート可視化。
- `update_system.py`: 収集から学習・推論 smoke test までの導線。

## 実装判断
- walk-forward fold は「初期 train 60% 以上、test window 固定、そこから等間隔に最大 5 fold」を採用している。
- `Model (LSTM)` / `Baselines` の旧キーは残しつつ、新スキーマは `legacy_holdout` と `walk_forward` を追加して後方互換を保つ。
