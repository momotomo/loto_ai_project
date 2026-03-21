# ARCHITECTURE

## 全体フロー
1. `data_collector.py` が `thekyo.jp` の CSV を取得し、整合性チェック後に `data/*_raw.csv` と `data/*_processed.csv` を更新する。
2. `train_prob_model.py` が処理済み CSV から 4 つの dataset ルートを組み立てる。`legacy` は既存の tabular 特徴、`multihot` は draw ごとの multi-hot と番号レベルの `hit / frequency / gap` 特徴、`deepsets` は draw 内の当選番号集合に対して `number_norm / frequency / gap` を element feature として持ち shared encoder -> mean pooling で draw embedding を生成する。`settransformer` は `deepsets` と同一の element feature を使いつつ、mean pooling 前に自己注意 (Set Attention Block) を挟んで集合内要素の相互作用を表現する。
3. 同スクリプトが leak-free な scaler と LSTM で `legacy_holdout` / `walk_forward` を評価し、fold の train 末尾だけを使った post-hoc calibration も variant ごとに比較する。
4. `best static baseline`・`legacy`・`multihot`・`deepsets`・`settransformer` 間の logloss 差に対して bootstrap CI と paired permutation test を計算し、calibration guardrail も含む `decision_summary` を生成する。
5. 同スクリプトが `data/eval_report_*.json`、`data/manifest_*.json`、`models/*_prob.keras`、`models/*_scaler.pkl`、必要時の `*_calibrator.json`、`*_feature_cols.json` を生成し、data hash / seed / preset / model variant / calibration method / Python / dependency versions も保存する。
6. `app.py` が成果物を読み、saved variant と saved calibration に応じて入力再構築と calibrator 適用を切り替えながら、確率可視化・買い目生成・評価レポート・manifest を表示する。
7. `update_system.py` は 1 → 2 → 3 → `predict.py` の smoke test を順に実行する。開発 CI では `--skip_data_refresh` を使って network 依存を避けられる。
8. `scripts/run_experiment.py` は既存パイプラインを呼び出し、run ごとの config / source hash / artifact snapshot を `runs/` へ残す。

## 主なファイル責務
- `config.py`: 宝くじごとの定数、lookback、サンプリング補助。
- `data_collector.py`: ダウンロード、retry、バリデーション、更新検知、特徴量付与。
- `train_prob_model.py`: 評価と本番学習、baseline 比較、artifact manifest 生成。
- `model_variants.py`: `legacy` / `multihot` / `deepsets` / `settransformer` dataset 構築、shared scaler 用の reshape helper、recent input 再構築、manifest 由来の variant 解決。
- `model_layers.py`: カスタム Keras レイヤー (`SetAttentionBlock`) の定義と `@register_keras_serializable` 登録。`train_prob_model.py` / `predict.py` / `app.py` がこれを import することで `load_model` が custom_objects 不要で動作する。
- `calibration_utils.py`: temperature / isotonic の fit / apply、ECE / reliability bins、calibrator artifact helper。
- `evaluation_statistics.py`: bootstrap CI と paired permutation test。
- `predict.py`: 保存済み成果物を使った簡易 CLI 推論。
- `app.py`: Streamlit UI、Kaggle 同期、評価レポート可視化。
- `update_system.py`: 収集から学習・推論 smoke test までの導線。
- `artifact_utils.py`: runtime version、file hash、prediction artifact integrity の共通 helper。
- `report_utils.py`: UI で使う variant / calibration / statistical test の互換 helper。
- `scripts/run_experiment.py` / `experiment_runner.py`: run 単位の実験追跡と artifact snapshot。

## 実装判断
- walk-forward fold は「初期 train 60% 以上、test window 固定、そこから等間隔に最大 5 fold」を採用している。
- `Model (LSTM)` / `Baselines` の旧キーは残しつつ、新スキーマは `legacy_holdout` と `walk_forward` を追加して後方互換を保つ。
- production artifact の既定経路は `legacy` のまま維持し、`multihot` / `deepsets` / `settransformer` は `evaluation_model_variants` に含めて比較する。
- calibration は `saved_calibration_method` で opt-in、`evaluation_calibration_methods` で比較し、production 保存と評価比較を分離する。
- `deepsets` は draw ごとに shared encoder -> mean pooling で集合 embedding を作り、その埋め込み系列を LSTM head に渡す軽量な Deep Sets sequence encoder として実装する。
- `settransformer` は `deepsets` と同一入力形状 `(batch, lookback, pick_count, element_features)` を受け取り、Dense 投影 → SetAttentionBlock (SAB, 2 head, key_dim=8) → mean pool → LSTM head という軽量アーキテクチャを採用する。element feature は `deepsets` と共通なので、attention の有無による効果差を公平に比較できる。
- `SetAttentionBlock` は `model_layers.py` に定義し `@register_keras_serializable(package="loto_ai")` で登録する。これにより `load_model` は custom_objects を明示しなくても動作する。
- 採用判定は challenger ごとに `variant_vs_best_static` と `variant_vs_legacy` の統計条件を確認し、さらに選択 calibration 後の logloss / Brier / ECE guardrail を満たしたものだけ昇格候補にする。`deepsets_vs_multihot` / `settransformer_vs_multihot` / `settransformer_vs_deepsets` は補助比較として保持する。
- `artifact_schema_version` は 3 を維持し、metadata 追加は additive change として扱う。個別 schema の進化は `manifest.schema_version` と `eval_report.schema_version` で表現する。
- `smoke` preset は学習性能比較ではなく plumbing 検証用に寄せ、0-epoch 構成で CI とローカル smoke を安定させる。
