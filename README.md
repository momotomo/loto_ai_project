# loto_ai_project

宝くじ予測の差は極小という前提で、確率出力そのものより「モデル設計の妥当性」と「更新判断の客観性」を重視するプロジェクトです。現行は `legacy` の tabular LSTM を保守的な既定経路として残しつつ、draw を multi-hot 時系列として扱う `multihot`、draw 内番号集合を shared encoder -> mean pooling で扱う `deepsets`、さらに attention で集合内要素の相互作用を表現する `settransformer` variant を比較します。評価は proper scoring rule・統計的検定・post-hoc calibration 比較で行います。主な成果物は `eval_report_*.json`、`manifest_*.json`、`prediction_history_*.json`、`*_prob.keras`、`*_scaler.pkl`、必要時の `*_calibrator.json` です。

## 主要コマンド
- `python data_collector.py --loto_type loto6`
- `python train_prob_model.py --loto_type loto6 --model_variant legacy --evaluation_model_variants legacy,multihot,deepsets,settransformer`
- `python train_prob_model.py --loto_type loto6 --model_variant legacy --evaluation_model_variants legacy,multihot,deepsets,settransformer --saved_calibration_method none --evaluation_calibration_methods none,temperature,isotonic`
- `python train_prob_model.py --loto_type loto6 --preset smoke --model_variant deepsets --evaluation_model_variants legacy,multihot,deepsets,settransformer --skip_final_train`
- `python train_prob_model.py --loto_type loto6 --preset smoke --model_variant legacy --evaluation_model_variants legacy,deepsets,settransformer --skip_final_train`
- `python scripts/run_experiment.py --config-json '{"loto_type":"loto6","preset":"smoke","seed":42,"model_variant":"deepsets","evaluation_model_variants":"legacy,multihot,deepsets,settransformer","refresh_data":false,"skip_final_train":true}'`
- `python update_system.py --loto_type loto6 --train_preset smoke --model_variant legacy --evaluation_model_variants legacy,multihot,deepsets,settransformer --skip_data_refresh`
- `streamlit run app.py`

## モデル variant
- `legacy`: `num1..numN` と既存集計特徴をそのまま tabular に並べた従来ルート。既定の保存 artifact は当面これを使います。
- `multihot`: 1 draw を `max_num` 次元 multi-hot とし、番号ごとの `hit / frequency / gap` を各 timestep に持つ新ルートです。
- `deepsets`: 1 draw を「当選番号の集合」として扱い、各番号 element に `number_norm / frequency / gap` を持たせ、shared encoder と mean pooling で draw embedding を作ってから lookback 系列を統合します。
- `settransformer`: `deepsets` と同一の element feature (`number_norm / frequency / gap`) を使いつつ、mean pooling 前に SetAttentionBlock (SAB: Multi-Head Self-Attention, 2 heads, key_dim=8) で集合内要素の相互作用を付加します。deepsets との唯一の構造差分が attention 有無なので公平な比較が可能です。カスタムレイヤー (`model_layers.py`) として `@register_keras_serializable` 登録済みで `load_model` が custom_objects 不要で動作します。
- calibration は opt-in で、`temperature` と `isotonic` を `evaluation_calibration_methods` に含めて比較し、保存する production artifact は `saved_calibration_method` で別管理します。
- `eval_report_*.json` には各 variant の `walk_forward` 結果、bootstrap CI、paired permutation test、pre/post calibration 指標、採用判定 summary を保存します。

## 依存と run tracking
- `requirements.txt` は direct dependency を固定し、`requirements.lock` は freeze ベースの lock として CI で使います。
- `scripts/run_experiment.py` は run ごとに `runs/` へ config・source hash・artifact copy・artifact hash を保存します。
- `manifest_*.json` と `eval_report_*.json` には `data_fingerprint`、`training_context`、`runtime_environment` が入り、data hash / preprocessing version / seed / preset / model variant / calibration method / Python / dependency version を確認できます。
- `smoke` preset は plumbing 確認用の 0-epoch 構成です。

## Walk-forward 評価とは
単発の holdout ではなく、時系列順を保ったまま train 区間を広げ、固定長の test window を複数回ずらして検証する方法です。このリポジトリでは各 fold ごとに scaler を train 期間のみで fit し直し、fold 単位の指標と平均・分散を `eval_report_*.json` に保存します。

## Static / Online baseline の違い
`static_baselines` は train 期間だけで作った予測を test 全体で固定比較する土俵です。`online_baselines` は test の実測を 1 ステップずつ取り込んで状態更新する参考値で、モデルの主比較対象ではありません。

## 更新判断
- 更新判断は「challenger variant が best static baseline と legacy の両方に対して、logloss 差の bootstrap CI 上限が 0 未満、かつ permutation test の p-value が閾値未満」であり、さらに選択 calibration 後の logloss / Brier が悪化せず、ECE guardrail を満たすときだけ昇格候補とします。
- `decision_summary` は `manifest` / `eval_report` / Streamlit evaluation tab に表示されます。

## 回別の予測照合
`prediction_history_*.json` には holdout / walk-forward の各評価対象 draw ごとの予測上位番号と実当選番号の照合結果を保存します。各 record には `model_variant` と `calibration_method` も入り、Streamlit の「✅ 実績との照合」タブで variant / calibration ごとに一致数の分布や回別 hit を確認できます。

## Streamlit 運用メモ
- Kaggle 同期欄の入力は `Kernel Ref (owner/kernel-slug)` 形式です。例: `username/my-loto-kernel`
- Kaggle 学習中の生成物は一度 `/kaggle/working/app/...` に作られますが、終了時に `/kaggle/working/data` と `/kaggle/working/models` へ export する前提です。同期側は root を正とし、移行期間だけ `app/` 配下も fallback で参照します。
- Streamlit Cloud では `/tmp` と app の `data/` / `models/` が別デバイスになることがあります。Kaggle 同期の staged sync は rename ではなく copy-based 配置で反映します。
- `processed.csv` と `feature_cols.json` / `scaler.pkl` / `model.keras` の世代がずれると、予測タブは整合性エラーで停止します。`multihot` / `deepsets` では manifest に記録された `model_variant` / `feature_strategy` / `input_summary` を使って入力を再構築します。
- その場合でも評価タブと実績照合タブは見られるようにしているので、manifest や prediction history を先に確認できます。
- GitHub Actions の翌営業日実行では、対象 loto_type だけ学習されます。Kaggle 同期も loto_type ごとに bundle 完全性を判定し、完全なものだけ部分更新します。
- `manifest_{loto_type}.json` には `artifact_schema_version` と `bundle_id` が入ります。同期はこの bundle 単位で検証し、新 bundle を temp copy で準備してから置換し、最後に不要な古い artifact だけ掃除します。
- 新しい manifest では `data_fingerprint` / `training_context` / `runtime_environment` / `artifact_metadata` も確認できます。

## Docs
- `AGENT.md`
- `docs/ARCHITECTURE.md`
- `docs/EVALUATION.md`
- `docs/DATA.md`
- `docs/ARTIFACTS.md`
- `docs/RUNBOOK.md`
- `docs/PR_SUMMARY.md`
- `docs/KAGGLE_GHACTIONS.md`
