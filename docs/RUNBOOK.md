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

## GitHub Actions からの Kaggle kick 失敗
- 症状: `.github/workflows/kaggle_kick.yml` が skip 以外で失敗する。
- 確認:
  - Secret `KAGGLE_JSON`
  - Variable `KAGGLE_KERNEL_ID`
  - Kaggle Notebook の Internet 設定
  - `docs/KAGGLE_GHACTIONS.md` の cron / setup 手順
  - `scripts/compute_kick_targets.py` が出した `targets` が想定どおりか
- 復旧:
  - まず `config-and-targets` job が ready=true になっているか確認する。
  - `config-and-targets` job の `targets` が空なら正常 skip。翌営業日条件に当たるか確認する。
  - `Prepare kernel build directory` の後に出る `script.py size_bytes=` と `run_config.json` の内容が想定どおりか確認する。
  - `kernel_status` が `failed/error/cancelled` の場合は Kaggle Notebook のログを確認する。
  - Kaggle schedule は停止し、Actions 側だけを実行源にする。

## Kaggle で file not found が出る
- 症状: Kaggle ログで `data_collector.py` や `train_prob_model.py` が見つからない。
- 確認:
  - workflow の `Debug build directory` ステップで `kernel-metadata.json` の `code_file` が `script.py` を指す前提になっているか
  - `script.py size_bytes=` が極端に小さくないか
  - build dir の `run_config.json` が対象ロトと preset を正しく持っているか
  - Kaggle ログの `[kaggle-entry] listing for /kaggle/working/app` に `data_collector.py` / `train_prob_model.py` / `run_config.json` が出ているか
- 復旧:
  - `scripts/kaggle_prepare_kernel_dir.py` の payload allowlist に必要ファイルが含まれているか見直す。
  - `scripts/kaggle_entry.py` が `/kaggle/working/app` へ payload を展開できているか確認する。
  - `Missing extracted payload files:` が出ている場合は、そのファイルが zip payload に入っていない。

## 翌営業日ルールの確認
- 祝日判定は内閣府 CSV `https://www8.cao.go.jp/chosei/shukujitsu/syukujitsu.csv` を使う。
- 営業日は平日かつ祝日でない日。
- GitHub Actions の cron は UTC なので、`09:10 JST` は `cron: '10 0 * * 1-5'` で表現する。
- 平常時の対象:
  - 月曜: `loto7`
  - 火曜: `loto6`
  - 水曜: `miniloto`
  - 金曜: `loto6`
- 祝日がある週は、抽せん日の翌営業日まで後ろ倒しになる。
- ローカル確認例:
  - `python scripts/compute_kick_targets.py --today 2026-03-11`

## 今回の実装判断
- GitHub Actions の schedule は広めに「平日朝」で起動し、実際に kick するかは `compute_kick_targets.py` 側で決めるようにした。
- 祝日考慮は外部ライブラリを足さず、内閣府の祝日 CSV を直接読む方式にした。
- Kaggle script kernel では `script.py` 単体しか見えない前提に切り替え、allowlist を zip payload にして `script.py` へ埋め込む方式にした。
- build dir にある `run_config.json` は workflow デバッグ用の見える化で、Kaggle 実行時は `script.py` 内の埋め込み payload から展開された `run_config.json` を使う。
- 対象ロトだけを `run_config.json` で渡し、Kaggle ではその target だけ `data_collector.py` と `train_prob_model.py` を回すようにした。
