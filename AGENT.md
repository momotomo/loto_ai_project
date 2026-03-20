# AGENT.md

## 1. プロジェクトの目的
- このプロジェクトは「宝くじはランダムであり、恒常的な優位性はほぼ期待できない」という前提で運用する。
- したがって最優先は精度の誇張ではなく、時系列リークを避けた評価と、再現可能な検証の信用度を上げること。
- `legacy` tabular ルートは保守的な既定経路として残し、`multihot` ルートは比較 artifact と統計的検定で評価してから昇格判断する。
- calibration は post-hoc / opt-in とし、`temperature` と `isotonic` を比較できるが、保存する production calibrator は `saved_calibration_method` で明示する。
- 予測 UI は娯楽用途であり、評価レポートと manifest を必ず確認してから判断する。

## 2. 無料枠前提の運用
- 学習はローカルまたは Kaggle Notebook の無料枠を前提とする。
- 推論 UI は `streamlit run app.py` でローカル起動する。
- 生成物は主に `data/` と `models/` に出力し、Kaggle 同期では CSV/JSON を `data/`、モデル本体を `models/` に置く。
- `eval_report_*.json` と `manifest_*.json` は軽量成果物、`*_prob.keras` と `*_scaler.pkl` は再生成可能な成果物として扱う。

## 3. コーディング規約
- 依存は最小限に保ち、標準ライブラリと既存依存を優先する。
- 時系列リークを禁止する。特に scaler は train 期間のみで fit し、test では transform のみを使う。
- 番号ごとの `frequency` / `gap` 系特徴は causal に構築し、prefix を伸ばしても過去行の値が変わらないことを前提にする。
- calibration fit も train 末尾から切り出した calibration split だけで行い、test draw を見ない。
- 再現性のため乱数 seed を固定し、主要評価設定は JSON/Docs に残す。
- ログは「何を取得したか」「何を生成したか」「更新があったか」「どこで失敗したか」が分かる粒度で出す。

## 4. 主要コマンド
- データ収集: `python data_collector.py --loto_type loto6`
- 全種類のデータ収集: `python data_collector.py`
- 学習と評価: `python train_prob_model.py --loto_type loto6 --model_variant legacy --evaluation_model_variants legacy,multihot`
- calibration 比較つき学習: `python train_prob_model.py --loto_type loto6 --model_variant legacy --evaluation_model_variants legacy,multihot --saved_calibration_method none --evaluation_calibration_methods none,temperature,isotonic`
- 実験追跡つき実行: `python scripts/run_experiment.py --config-json '{"loto_type":"loto6","preset":"smoke","seed":42,"model_variant":"multihot","evaluation_model_variants":"legacy,multihot","refresh_data":false,"skip_final_train":true}'`
- UI 起動: `streamlit run app.py`
- 全自動更新: `python update_system.py --loto_type loto6 --train_preset smoke --model_variant legacy --evaluation_model_variants legacy,multihot --skip_data_refresh`

## 5. テスト方針
- 最低限 `python -m py_compile $(rg --files -g '*.py' -g '!venv/**')` を通す。
- `pytest -q` を追加し、validation / leakage / multi-hot dataset / 統計検定 / artifact integrity / run tracking / CLI smoke の退行を先に止める。
- 実行チェックは `data_collector.py`、`train_prob_model.py`、`predict.py`、`update_system.py` の簡易 smoke test を優先する。
- ネットワーク依存の取得は、失敗してもメッセージで原因が追えることを確認する。

## 6. PR 方針
- コミットは小さめに分け、学習ロジック、UI、Docs を混ぜすぎない。
- 評価 JSON と生成物仕様は後方互換を意識し、既存 UI/運用を壊さない。
- 実装判断が暗黙になりそうなら README / `docs/` / コメントのいずれかに理由を残す。
- 新しい成果物や評価指標を追加した場合は、対応する Docs を同時に更新する。
