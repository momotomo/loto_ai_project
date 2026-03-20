# KAGGLE GITHUB ACTIONS

## 目的
- GitHub Actions の `schedule` / `workflow_dispatch` から Kaggle Notebook を push して実行する。
- Kaggle 側の schedule は止め、時刻制御は GitHub Actions 側へ寄せる前提にする。
- 実行対象は「抽せん日の翌営業日」に該当するロト種別だけに絞り、Kaggle 側の計算量を最小化する。

## 追加ファイル
- `.github/workflows/kaggle_kick.yml`
- `scripts/kaggle_prepare_kernel_dir.py`
- `scripts/kaggle_entry.py`
- `scripts/compute_kick_targets.py`
- `artifact_utils.py`

## セットアップ
1. Kaggle で API token を発行し、`kaggle.json` の中身を GitHub Secret `KAGGLE_JSON` に登録する。
2. Repository Variable `KAGGLE_KERNEL_ID` に `username/kernel-slug` を登録する。
3. 任意で `KAGGLE_KERNEL_TITLE` を登録する。未設定なら slug から自動生成する。
4. Kaggle Notebook 側では Internet を有効にし、既存の Kaggle schedule は無効化する。

## ワークフローの挙動
- Secrets / Vars が未設定なら `config-and-targets` job だけ成功し、本体 job は skip される。
- `scripts/compute_kick_targets.py` が JST の今日を基準に「抽せん日の翌営業日」対象を計算する。
- 対象が空なら workflow 全体は成功終了し、Kaggle kick は skip される。
- 設定済みかつ対象ありなら build dir を生成し、`kaggle kernels push -p <build_dir>` で新しい実行を開始する。
- その後は Kaggle API で status をポーリングし、`complete` で成功、`failed/error/cancelled` で失敗にする。
- `Debug build directory` ステップで `script.py` のサイズと `run_config.json` の内容を出す。push 前の payload 生成結果を確認するためのログで、Kaggle 実行時には `script.py` 内の埋め込み payload が使われる。

## Kaggle 上での実行内容
- Kaggle script kernel では `code_file` 単体しか確実に見えない前提にする。`/kaggle/src` に複数ファイルが並ぶことを期待しない。
- `scripts/kaggle_prepare_kernel_dir.py` は allowlist の Python ファイルと `data/*.csv` と `run_config.json` を zip 化し、base64 payload として `script.py` に埋め込む。
- `scripts/kaggle_entry.py` は runtime テンプレートとして使い、起動時に payload を `/kaggle/working/app` へ展開してから、そのディレクトリを `cwd` にして実行する。
- `run_config.json` に入っている targets だけを順に処理し、各 target ごとに `data_collector.py --loto_type X` と `train_prob_model.py --loto_type X --preset fast --seed 42 --skip_legacy_holdout` を実行する。
- 取得失敗時は build dir に同梱した `data/` を fallback として使い、学習を継続する。
- 学習・推論で作られた成果物はまず `/kaggle/working/app/data` と `/kaggle/working/app/models` に置かれ、終了時に `/kaggle/working/data` と `/kaggle/working/models` へ export する。Streamlit 同期は root を正とし、移行期間だけ `app/` 配下も fallback で読む。
- payload 展開方式にした理由は、Kaggle 側で `/kaggle/src` に `script.py` しか存在せず `data_collector.py` / `train_prob_model.py` が見つからない失敗を根本的に避けるため。
- `artifact_utils.py` も payload に含め、manifest / eval_report の hash と runtime metadata が Kaggle 側でも同じコードで生成されるようにする。

## 翌営業日判定
- 祝日判定は内閣府 CSV `https://www8.cao.go.jp/chosei/shukujitsu/syukujitsu.csv` を参照する。
- 営業日は「平日かつ祝日でない日」とする。
- 各ロトの抽せん曜日:
  - `miniloto`: 火曜
  - `loto6`: 月曜 / 木曜
  - `loto7`: 金曜
- 今日より前の直近抽せん日 `last_draw` を求め、その翌営業日が今日なら対象に含める。
- 平常時の例:
  - 水曜: `miniloto`
  - 火曜 / 金曜: `loto6`
  - 月曜: `loto7`
  - 祝日が挟まる場合は次の営業日に後ろ倒しされる。

## cron と UTC/JST
- GitHub Actions の cron は UTC。
- この workflow は平日 `09:10 JST` に相当する `cron: '10 0 * * 1-5'` を使う。
- 休日・祝日は workflow が動いても `compute_kick_targets.py` 側で対象 0 件になり、Kaggle kick を skip する。
- 手動確認したい場合は `workflow_dispatch` の `today_jst` に任意の日付を入れる。

## よくある失敗
- `KAGGLE_JSON` が壊れている:
  - Secret に `kaggle.json` の JSON 本文をそのまま入れる。
- `KAGGLE_KERNEL_ID` が違う:
  - `username/kernel-slug` 形式で登録する。
- Kaggle 側で Internet が無効:
  - `data_collector.py` が live 取得できない。fallback は効くが最新化はできない。
- 既存 Kaggle schedule と二重実行:
  - Kaggle 側 schedule を止め、GitHub Actions 側だけ残す。
- Kaggle で `file not found`:
  - workflow の `Debug build directory` ステップで `script.py` のサイズが極端に小さくないか、`run_config.json` が期待どおりか確認する。
  - Kaggle ログの `[kaggle-entry] listing for /kaggle/working/app` に、展開後の `data_collector.py` / `train_prob_model.py` / `run_config.json` が出ているか確認する。
  - 学習完了後は `[kaggle-entry] export data:` / `[kaggle-entry] export models:` のログと、`/kaggle/working/data` / `/kaggle/working/models` の listing が出ているか確認する。
  - payload 展開先に必要ファイルが無い場合は `Missing extracted payload files:` のログに欠落ファイル名が出る。
