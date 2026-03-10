# loto_ai_project

宝くじ予測の差は極小という前提で、LSTM による確率ベクトル出力と、信用できる時系列評価を重視するプロジェクトです。主な成果物は `eval_report_*.json`、`manifest_*.json`、`prediction_history_*.json`、`*_prob.keras`、`*_scaler.pkl` です。

## 主要コマンド
- `python data_collector.py --loto_type loto6`
- `python train_prob_model.py --loto_type loto6`
- `python update_system.py`
- `streamlit run app.py`

## Walk-forward 評価とは
単発の holdout ではなく、時系列順を保ったまま train 区間を広げ、固定長の test window を複数回ずらして検証する方法です。このリポジトリでは各 fold ごとに scaler を train 期間のみで fit し直し、fold 単位の指標と平均・分散を `eval_report_*.json` に保存します。

## Static / Online baseline の違い
`static_baselines` は train 期間だけで作った予測を test 全体で固定比較する土俵です。`online_baselines` は test の実測を 1 ステップずつ取り込んで状態更新する参考値で、モデルの主比較対象ではありません。

## 回別の予測照合
`prediction_history_*.json` には holdout / walk-forward の各評価対象 draw ごとの予測上位番号と実当選番号の照合結果を保存します。Streamlit の「✅ 実績との照合」タブで、一致数の分布や回別の hit を確認できます。

## Streamlit 運用メモ
- Kaggle 同期欄の入力は `Kernel Ref (owner/kernel-slug)` 形式です。例: `username/my-loto-kernel`
- `processed.csv` と `feature_cols.json` / `scaler.pkl` / `model.keras` の世代がずれると、予測タブは整合性エラーで停止します。
- その場合でも評価タブと実績照合タブは見られるようにしているので、manifest や prediction history を先に確認できます。
- GitHub Actions の翌営業日実行では、対象 loto_type だけ学習されます。Kaggle 同期も loto_type ごとに bundle 完全性を判定し、完全なものだけ部分更新します。

## Docs
- `AGENT.md`
- `docs/ARCHITECTURE.md`
- `docs/EVALUATION.md`
- `docs/DATA.md`
- `docs/ARTIFACTS.md`
- `docs/RUNBOOK.md`
- `docs/PR_SUMMARY.md`
- `docs/KAGGLE_GHACTIONS.md`
