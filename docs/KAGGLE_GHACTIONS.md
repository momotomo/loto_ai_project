# KAGGLE GITHUB ACTIONS

## 目的
- GitHub Actions の `schedule` / `workflow_dispatch` から Kaggle Notebook を push して実行する。
- Kaggle 側の schedule は止め、時刻制御は GitHub Actions 側へ寄せる前提にする。

## 追加ファイル
- `.github/workflows/kaggle_kick.yml`
- `scripts/kaggle_prepare_kernel_dir.py`
- `scripts/kaggle_entry.py`

## セットアップ
1. Kaggle で API token を発行し、`kaggle.json` の中身を GitHub Secret `KAGGLE_JSON` に登録する。
2. Repository Variable `KAGGLE_KERNEL_ID` に `username/kernel-slug` を登録する。
3. 任意で `KAGGLE_KERNEL_TITLE` を登録する。未設定なら slug から自動生成する。
4. Kaggle Notebook 側では Internet を有効にし、既存の Kaggle schedule は無効化する。

## ワークフローの挙動
- Secrets / Vars が未設定なら `config-check` job だけ成功し、本体 job は skip される。
- 設定済みなら build dir を生成し、`kaggle kernels push -p <build_dir>` で新しい実行を開始する。
- その後は Kaggle API で status をポーリングし、`complete` で成功、`failed/error/cancelled` で失敗にする。

## Kaggle 上での実行内容
- `scripts/kaggle_entry.py` が `data_collector.py` を先に試す。
- 取得失敗時は build dir に同梱した `data/` を fallback として使い、`train_prob_model.py` と `predict.py` を継続する。
- 成果物は `/kaggle/working` 配下に残るので、Kaggle Output から同期できる。

## cron と JST
- GitHub Actions の cron は UTC。
- 例: 毎日 00:15 JST に回したい場合、UTC では前日 15:15 なので `cron: '15 15 * * *'`。
- 今回の workflow はこの例をデフォルトにしている。

## よくある失敗
- `KAGGLE_JSON` が壊れている:
  - Secret に `kaggle.json` の JSON 本文をそのまま入れる。
- `KAGGLE_KERNEL_ID` が違う:
  - `username/kernel-slug` 形式で登録する。
- Kaggle 側で Internet が無効:
  - `data_collector.py` が live 取得できない。fallback は効くが最新化はできない。
- 既存 Kaggle schedule と二重実行:
  - Kaggle 側 schedule を止め、GitHub Actions 側だけ残す。
