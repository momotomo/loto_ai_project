# AGENT.md

## 1. プロジェクトの目的
- このプロジェクトは「宝くじはランダムであり、恒常的な優位性はほぼ期待できない」という前提で運用する。
- したがって最優先は精度の誇張ではなく、時系列リークを避けた評価と、再現可能な検証の信用度を上げること。
- 予測 UI は娯楽用途であり、評価レポートと manifest を必ず確認してから判断する。

## 2. 無料枠前提の運用
- 学習はローカルまたは Kaggle Notebook の無料枠を前提とする。
- 推論 UI は `streamlit run app.py` でローカル起動する。
- 生成物は主に `data/` と `models/` に出力し、Kaggle 同期では CSV/JSON を `data/`、モデル本体を `models/` に置く。
- `eval_report_*.json` と `manifest_*.json` は軽量成果物、`*_prob.keras` と `*_scaler.pkl` は再生成可能な成果物として扱う。

## 3. コーディング規約
- 依存は最小限に保ち、標準ライブラリと既存依存を優先する。
- 時系列リークを禁止する。特に scaler は train 期間のみで fit し、test では transform のみを使う。
- 再現性のため乱数 seed を固定し、主要評価設定は JSON/Docs に残す。
- ログは「何を取得したか」「何を生成したか」「更新があったか」「どこで失敗したか」が分かる粒度で出す。

## 4. 主要コマンド
- データ収集: `python data_collector.py --loto_type loto6`
- 全種類のデータ収集: `python data_collector.py`
- 学習と評価: `python train_prob_model.py --loto_type loto6`
- UI 起動: `streamlit run app.py`
- 全自動更新: `python update_system.py`

## 5. テスト方針
- 最低限 `python -m py_compile $(rg --files -g '*.py' -g '!venv/**')` を通す。
- 実行チェックは `data_collector.py`、`train_prob_model.py`、`predict.py` の簡易 smoke test を優先する。
- ネットワーク依存の取得は、失敗してもメッセージで原因が追えることを確認する。

## 6. PR 方針
- コミットは小さめに分け、学習ロジック、UI、Docs を混ぜすぎない。
- 評価 JSON と生成物仕様は後方互換を意識し、既存 UI/運用を壊さない。
- 実装判断が暗黙になりそうなら README / `docs/` / コメントのいずれかに理由を残す。
- 新しい成果物や評価指標を追加した場合は、対応する Docs を同時に更新する。
