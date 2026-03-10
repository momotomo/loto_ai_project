# RUNBOOK

## 取得失敗
- 症状: `data_collector.py` が `CSV 取得に失敗しました` で終了する。
- 確認:
  - 対象 URL が生きているか
  - ネットワーク制限がないか
  - `thekyo.jp` 側の応答形式が変わっていないか
- 復旧:
  - まず `python data_collector.py --loto_type loto6` を単体実行してエラー内容を確認する。
  - CSV 構造が変わった場合は `parse_downloaded_csv()` の列切り出しを更新する。

## Kaggle 同期失敗
- 症状: Streamlit の同期ボタンで認証または取得エラーになる。
- 確認:
  - `KAGGLE_USERNAME`, `KAGGLE_KEY`, `KAGGLE_SLUG`
  - Notebook Output に `eval_report_*.json` / `manifest_*.json` / `*_prob.keras` があるか
- 復旧:
  - Notebook 側で学習を再実行し、Output が更新されたことを確認してから再同期する。

## モデル読み込み失敗
- 症状: `app.py` または `predict.py` で必要ファイル不足と表示される。
- 確認:
  - `models/{loto_type}_prob.keras`
  - `models/{loto_type}_scaler.pkl`
  - `data/{loto_type}_feature_cols.json` または `models/{loto_type}_feature_cols.json`
- 復旧:
  - `python train_prob_model.py --loto_type loto6` を再実行する。
  - Kaggle 利用時は同期をやり直す。
  - 評価だけ先に見たい場合は `venv/bin/python train_prob_model.py --loto_type loto6 --preset smoke --skip_final_train` を使い、運用モデル更新は後で本実行する。

## 学習は通るが評価が不自然
- 症状: static baseline より極端に良すぎる、または calibration が不自然。
- 確認:
  - scaler が train only fit になっているか
  - fold の train/test 境界が時系列順か
  - `walk_forward.aggregate` の平均と fold 個別成績が整合しているか
  - leak-free 評価と final の全データ fit を混同していないか
- 復旧:
  - `docs/EVALUATION.md` のリーク防止項目に沿って split 実装を見直す。

## CPU が重くて回らない
- 症状: ローカル無料 CPU で full 学習が長い。
- 対処:
  - `venv/bin/python train_prob_model.py --loto_type loto6 --preset smoke`
  - `venv/bin/python train_prob_model.py --loto_type loto6 --preset fast`
  - `venv/bin/python update_system.py --loto_type loto6 --train_preset smoke`
- メモ:
  - `legacy_holdout` / `walk_forward` は leak-free 評価。
  - final model は運用用の全データ fit。
  - `--skip_final_train` は既存運用成果物を再利用し、互換成果物が無い場合だけ 0-epoch の雛形を保存する。

## update_system 失敗
- 症状: `update_system.py` 実行中に途中で停止する。
- 復旧:
  - 失敗したステップのスクリプトを単体で実行する。
  - `data_collector.py` → `train_prob_model.py` → `predict.py` の順で個別に確認する。
