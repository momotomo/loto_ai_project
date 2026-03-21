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
  - 入力欄の `Kernel Ref (owner/kernel-slug)`
  - Notebook Output に `eval_report_*.json` / `manifest_*.json` / `*_prob.keras` があるか
- 復旧:
  - Notebook 側で学習を再実行し、Output が更新されたことを確認してから再同期する。
  - 翌営業日実行では対象 loto_type だけが学習対象なので、他 loto_type の `processed.csv` だけが残っていても即失敗ではない。
  - Output の path mismatch が疑わしい場合は、`/kaggle/working/data` / `models` へ export 済みか確認する。移行期間は Streamlit 側で `app/data` / `app/models` fallback も読むが、root export が正。
  - Streamlit Cloud で `[Errno 18] Invalid cross-device link` が出る場合は、`/tmp` の staged bundle から app 領域への rename が失敗している。artifact 不足は二次被害なので、copy-based 配置が入った版へ更新して再同期する。

## Streamlit の KeyError / 整合性エラー
- 症状: 予測タブで `df[feature_cols]` 付近が落ちる、または整合性エラーが表示される。
- 主な原因:
  - `processed.csv` と `feature_cols.json` の世代不一致
  - manifest の `model_variant` と model/scaler の入力次元不一致
  - 同期失敗後に古い `scaler.pkl` / `model.keras` が残っている
  - Kaggle 同期で一部 artifact だけ古いまま混在している
  - `bundle_id` の異なる成果物が混ざっている
- UI 上で見る場所:
  - 予測タブのエラーブロック
  - 評価タブの manifest 表示
  - サイドバー同期欄の最後の同期メッセージ
- 復旧:
  - サイドバーの `🧹 {loto_type} のローカル artifact を削除` を実行する
  - Kaggle 同期を再実行する
  - 必要に応じて `feature_cols.json` / `processed.csv` / `scaler.pkl` / `model.keras` を削除して再取得する
  - prediction tab は安全停止するが、評価タブと実績照合タブはそのまま確認してよい

## モデル読み込み失敗
- 症状: `app.py` または `predict.py` で必要ファイル不足と表示される。
- 確認:
  - `models/{loto_type}_prob.keras`
  - `models/{loto_type}_scaler.pkl`
  - `data/{loto_type}_feature_cols.json` または `models/{loto_type}_feature_cols.json`
  - `saved_calibration_method != none` の場合は `data/{loto_type}_calibrator.json` または `models/{loto_type}_calibrator.json`
- 復旧:
  - `python train_prob_model.py --loto_type loto6` を再実行する。
  - Kaggle 利用時は同期をやり直す。
  - 評価だけ先に見たい場合は `venv/bin/python train_prob_model.py --loto_type loto6 --preset smoke --model_variant legacy --evaluation_model_variants legacy,multihot,deepsets --skip_final_train` を使い、運用モデル更新は後で本実行する。

## prediction history が無い
- 症状: Streamlit の「✅ 実績との照合」で `prediction history が未生成です` と表示される。
- 確認:
  - `data/prediction_history_{loto_type}.json`
  - `data/manifest_{loto_type}.json` の `prediction_history_rows`
- 復旧:
  - `python train_prob_model.py --loto_type loto6 --preset smoke --eval_epochs 0 --final_epochs 0`
  - Kaggle 利用時は同期ボタンで最新 Output を取り直す。
  - `update_system.py` を使う場合は成果物確認で `prediction_history_*.json` もチェックされる。

## 学習は通るが評価が不自然
- 症状: static baseline より極端に良すぎる、または calibration が不自然。
- 確認:
  - scaler が train only fit になっているか
  - fold の train/test 境界が時系列順か
  - calibration split が train 末尾だけで構成されているか
  - `walk_forward.aggregate` の平均と fold 個別成績が整合しているか
  - leak-free 評価と final の全データ fit を混同していないか
- 復旧:
  - `docs/EVALUATION.md` のリーク防止項目に沿って split 実装を見直す。

## CPU が重くて回らない
- 症状: ローカル無料 CPU で full 学習が長い。
- 対処:
  - `venv/bin/python train_prob_model.py --loto_type loto6 --preset smoke --model_variant legacy --evaluation_model_variants legacy,multihot,deepsets`
  - `venv/bin/python train_prob_model.py --loto_type loto6 --preset fast`
  - `venv/bin/python update_system.py --loto_type loto6 --train_preset smoke --model_variant legacy --evaluation_model_variants legacy,multihot,deepsets --skip_data_refresh`
- メモ:
  - `legacy_holdout` / `walk_forward` は leak-free 評価。
  - final model は運用用の全データ fit。
  - `--skip_final_train` は既存運用成果物を再利用し、互換成果物が無い場合だけ 0-epoch の雛形を保存する。
  - `smoke` preset は 0-epoch の plumbing 確認用。

## update_system 失敗
- 症状: `update_system.py` 実行中に途中で停止する。
- 復旧:
  - 失敗したステップのスクリプトを単体で実行する。
  - `data_collector.py` → `train_prob_model.py` → `predict.py` の順で個別に確認する。
  - network 要因を切り分けたい場合は `venv/bin/python update_system.py --loto_type loto6 --train_preset smoke --skip_final_train --skip_data_refresh` を使う。
  - `skip_final_train` で再利用したい場合でも、model/scaler/feature_cols の入力次元が揃わないと再利用せず雛形へ切り替わる。variant を切り替えた直後はこの挙動が正常。

## 実験の再現 run を残したい
- 症状: どの config / source / artifact で実験したかを run 単位で残したい。
- 対処:
  - `venv/bin/python scripts/run_experiment.py --config-json '{"loto_type":"loto6","preset":"smoke","seed":42,"model_variant":"deepsets","evaluation_model_variants":"legacy,multihot,deepsets","refresh_data":false,"skip_final_train":true}'`
- 生成物:
  - `runs/<run_id>/config/*.json`
  - `runs/<run_id>/source_hashes.json`
  - `runs/<run_id>/run_summary.json`
  - `runs/<run_id>/artifacts/...`

## deepsets vs settransformer を比較したい
- 症状: smoke での 0-epoch 比較では優劣が分からない。ある程度学習させて公平に比較したい。
- 対処:
  1. `archcomp` preset で多 seed 比較を実行する:
     ```bash
     venv/bin/python scripts/run_multi_seed.py \
         --loto_type loto6 \
         --preset archcomp \
         --seeds 42,123,456 \
         --model_variant legacy \
         --evaluation_model_variants legacy,multihot,deepsets,settransformer \
         --saved_calibration_method none \
         --evaluation_calibration_methods none,temperature,isotonic \
         --run_root runs
     ```
  2. 生成された `data/comparison_summary_loto6.json` を確認する:
     - `variants.deepsets.logloss.mean` vs `variants.settransformer.logloss.mean`
     - `pairwise_comparisons.settransformer_vs_deepsets.both_pass_count`
     - `pairwise_comparisons.settransformer_vs_deepsets.ci_wins` / `permutation_wins`
  3. promote_count が 2/3 以上なら settransformer に対して本番切り替えを検討する。
- メモ:
  - `archcomp` preset は production 保存しない前提で設計している（`--skip_final_train` がデフォルト）。
  - production を変えたい場合は `--no_skip_final_train` を追加するか、通常の `train_prob_model.py` を使う。
  - 各 seed の run artifact は `runs/<run_id>/` に独立して残る。
  - comparison summary は `data/comparison_summary_{loto_type}.json` に上書きされる。
  - 次の variant (PMA / ISAB) を追加する前に、まずこの比較を実施することを推奨する。

## campaign を使って比較を継続監視したい

- 症状: 比較を 1 回で終わらせず、前回との差分と傾向の変化を追いたい。
- 対処:
  1. `scripts/run_campaign.py` で named campaign を実行する:
     ```bash
     # プロファイル一覧を確認
     python scripts/run_campaign.py --list_profiles

     # 標準 archcomp キャンペーンを実行（campaign_name は日付などで一意に）
     python scripts/run_campaign.py --campaign_name 2026-03-21_archcomp --profile archcomp

     # 軽量確認（fast preset, 2 seeds, loto6 のみ）
     python scripts/run_campaign.py --campaign_name 2026-03-21_lite --profile archcomp_lite

     # run_more_seeds が ≥3 回続く場合は archcomp_full
     python scripts/run_campaign.py --campaign_name 2026-03-21_full --profile archcomp_full
     ```
  2. **まず diff report を読む**（前回からの変化）:
     - `data/campaign_diff_report.md` — variant ranking / pairwise 変化・recommendation 変化
  3. campaign history で stability を確認する:
     - `data/campaign_history.json` → `recommendation_stability` を参照
     - `consecutive_same_action >= 3` かつ `run_more_seeds` → `archcomp_full` を実行
     - `consecutive_same_action >= 2` かつ `consider_promotion` → promotion を慎重に検討
  4. 今回 campaign の evidence pack を読む:
     - `campaigns/<campaign_name>/cross_loto_report.md` — 詳細 evidence pack
     - `campaigns/<campaign_name>/recommendation.json` — next_action

- profile ごとの使い分け:
  | profile | preset | seeds | loto_types | 用途 |
  |---------|--------|-------|-----------|------|
  | `archcomp_lite` | fast | 2 | loto6 のみ | sanity check のみ（決定に使わない） |
  | `archcomp` | archcomp | 3 | 全 3 種 | 標準 campaign（既定） |
  | `archcomp_full` | default | 5 | 全 3 種 | `run_more_seeds` が続く場合 |

- メモ:
  - campaign 間の artifact は `campaigns/<name>/` に独立保存される（上書きなし）
  - `--skip_final_train` が既定 True のため production artifact は変わらない
  - campaign_name が既存の場合はエラーで停止するので一意な名前を使うこと
  - history は `data/campaign_history.json` に蓄積される
  - campaign_diff_report.md は最新 2 campaign 間の差分のみ

## 複数 loto_type を横断して variant を比較したい（ad-hoc）

- 症状: loto6 だけで比較したが、miniloto / loto7 での傾向が不明。cross-loto で全体傾向を把握したい。
- 対処:
  1. `scripts/run_cross_loto.py` で全 loto_type 横断の比較を実行する:
     ```bash
     venv/bin/python scripts/run_cross_loto.py \
         --loto_types loto6,loto7,miniloto \
         --preset archcomp \
         --seeds 42,123,456 \
         --evaluation_model_variants legacy,multihot,deepsets,settransformer \
         --run_root runs
     ```
  2. 生成された artifact を確認する（**まず Markdown を読む**）:
     - `data/cross_loto_report.md` — 全情報を人が読める Markdown evidence pack (**読み始めはここ**)
     - `data/variant_metrics.csv` / `data/pairwise_comparisons.csv` / `data/recommendation_summary.csv` — 表計算用 CSV
     - `data/cross_loto_summary.json` — 横断 ranking / pairwise / promotion 傾向（raw JSON）
     - `data/recommendation.json` — 次に何をすべきかの推奨（raw JSON）
     - `data/comparison_summary_{loto_type}.json` — 各 loto_type の per-seed 集計（raw JSON）
  3. Markdown の `## Recommendation` 節で `recommended_next_action` を確認する:
     - `hold` → production を変えず追加実験を検討
     - `run_more_seeds` → seed 数を増やして信頼区間を絞る（`PAIRWISE_SIGNAL_THRESHOLD >= 0.5` の pair がある）
     - `consider_promotion` → 候補 variant の本番学習を検討（`CONSISTENT_PROMOTE_THRESHOLD >= 0.5` を満たした variant がある）
  4. `recommendation.whether_to_try_pma_or_isab_next` が true なら PMA / ISAB の試験を検討する。
     （条件: settransformer_vs_deepsets の both_pass_count/run_count ≥ 0.5）
- メモ:
  - 既存の `comparison_summary_{loto_type}.json` から学習をスキップして集計だけやり直せる:
    ```bash
    venv/bin/python scripts/run_cross_loto.py --loto_types loto6,loto7,miniloto --skip_training
    ```
  - 既存の `cross_loto_summary.json` / `recommendation.json` から Markdown + CSV だけ再生成できる:
    ```bash
    venv/bin/python scripts/run_cross_loto.py --report_only
    ```
  - 各 loto_type の per-seed run は `runs/<run_id>/` に独立して残る。
  - cross-loto summary は `data/cross_loto_summary.json` に上書きされる。
  - `--skip_final_train` が既定 True のため、比較実行で production artifact が上書きされることはない。
  - 新しい variant を追加する前に必ずこの cross-loto summary を確認すること。
  - **継続監視が目的なら `scripts/run_campaign.py` の使用を優先すること。**

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
  - `Poll kernel status` の `kernel_status_raw=` で CLI の生出力を確認する。
  - `kernel_status` が `failed/error/cancelled` の場合は Kaggle Notebook のログを確認する。
  - Kaggle schedule は停止し、Actions 側だけを実行源にする。
  - Kaggle notebook log 側が成功で、Actions の poll だけ落ちている場合は監視コードの問題を疑う。

## Kaggle 403 / 404 の切り分け
- 症状: Streamlit の Kaggle 同期で 403 または 404 が出る。
- 主な原因:
  - Kernel Ref が `owner/kernel-slug` ではなく slug 単体になっている
  - `KAGGLE_USERNAME` / `KAGGLE_KEY` に、その kernel を読める権限がない
  - private kernel に対して別アカウントの token を使っている
- UI 上で見る場所:
  - サイドバーの `Kernel Ref (owner/kernel-slug)` 入力欄
  - 同期失敗時のエラーメッセージ
- 復旧:
  - Kernel Ref を確認する。例: `username/my-loto-kernel`
  - Streamlit secrets の `KAGGLE_USERNAME` / `KAGGLE_KEY` を確認する
  - private kernel なら所有者アカウント、または権限付きアカウントの token に切り替える
  - 404 の場合は owner / slug の typo を疑う

## Kaggle 同期の部分更新
- GitHub Actions の翌営業日実行では、今回対象の loto_type だけ学習される。
- そのため Kaggle Output に全 loto_type の完全 bundle が常にあるとは限らない。
- Streamlit 同期は `kaggle_run_summary.json` → `run_config.json` → `manifest_*.json` の順で今回対象を推定し、loto_type ごとに完全 bundle を判定する。
- artifact 探索は root の `data/...` / `models/...` を優先し、旧 Output 互換のため `app/data/...` / `app/models/...` も fallback で参照する。
- bundle 判定は `manifest_{loto_type}.json` の `artifact_schema_version` / `bundle_id` と必須成果物の存在で行う。
- 更新対象 loto_type では、新 bundle の全ファイルを final dir 上の temp file へ copy してから置換し、最後に不要な古いローカル artifact だけを掃除する。
- local 反映は `/tmp` staged dir から final dir へ直接 rename せず、`data/` / `models/` 側に sibling temp を copy してから `os.replace()` する。Streamlit Cloud の cross-device 制約を避けるため。
- UI 上の表示:
  - `更新: loto6`
  - `loto6: bundle_id=... / source=root`
  - `スキップ: loto7（今回の実行対象外）`
  - `スキップ: miniloto（bundle 不完全）`
- 復旧:
  - 今回対象外のスキップは正常動作
  - 今回対象の loto_type が `bundle 不完全` なら、その loto_type の学習が途中で終わっていないか Kaggle Output を確認する
  - 全 loto_type がスキップされる場合だけ同期エラーになる

## Kaggle で file not found が出る
- 症状: Kaggle ログで `data_collector.py` や `train_prob_model.py` が見つからない。
- 確認:
  - workflow の `Debug build directory` ステップで `kernel-metadata.json` の `code_file` が `script.py` を指す前提になっているか
  - `script.py size_bytes=` が極端に小さくないか
  - build dir の `run_config.json` が対象ロトと preset / model_variant / evaluation_model_variants / evaluation_calibration_methods を正しく持っているか
  - Kaggle ログの `[kaggle-entry] listing for /kaggle/working/app` に `data_collector.py` / `train_prob_model.py` / `model_variants.py` / `evaluation_statistics.py` / `calibration_utils.py` / `report_utils.py` / `run_config.json` が出ているか
- 復旧:
  - `scripts/kaggle_prepare_kernel_dir.py` の payload allowlist に必要ファイルが含まれているか見直す。
  - `scripts/kaggle_entry.py` が `/kaggle/working/app` へ payload を展開できているか確認する。
  - 学習本体が通っているのに Streamlit 同期で artifact 不足になる場合は、`/kaggle/working/app/data` と `/kaggle/working/app/models` に生成された後、root の `/kaggle/working/data` と `/kaggle/working/models` に export されたか確認する。
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
- `Poll kernel status` の不具合原因は Kaggle Python API メソッド名の typo で、`kernel_status` ではなく `kernels_status` だった。今回は typo 依存を避けるため、監視を CLI の `kaggle kernels status` ベースへ寄せた。
- prediction history の正式な比較対象は `predicted_top_k` とし、サンプリング買い目履歴とは分離した。将来 live 予測を扱うときは `pending/resolved` の別 artifact に広げる前提にしている。
- Streamlit の整合性エラー時は app 全体を落とさず、予測タブだけ安全停止して評価タブと実績照合タブを残すようにした。artifact 世代の切り分けを UI 上で継続できる方を優先した。
- Kaggle 同期は一時ディレクトリへ全 artifact を集め、bundle が不完全ならローカル更新を中止するようにした。partial update で世代が混ざる事故を減らすため。
- ただし翌営業日実行では非対象 loto_type の不完全 bundle が混ざるため、同期判定は global ではなく loto_type 単位に変えた。対象外はスキップ表示に留め、対象 loto_type だけ厳密に守る方針にした。
- `artifact_schema_version` / `bundle_id` を manifest に入れ、clean sync はその bundle 単位で行うようにした。schema version 変更直後や初回移行時は `miniloto` / `loto6` / `loto7` を一度フル再学習して bundle を揃え、その後は partial update で運用する。
- 今回の Kaggle 同期不具合は path mismatch が原因で、学習後の artifact が `/kaggle/working/app/...` にだけ残り root の `data/` / `models/` と噛み合っていなかった。運用上は root export を正とし、Streamlit 側は旧 Output 互換として `app/` 配下も fallback 参照する。
- 今回の cross-device sync failure は `/tmp` staged dir から app 領域へ `os.replace()` したことが原因だった。現在は final dir 上の temp file へ copy してから replace する方式に変え、copy 完了前に local bundle を消さない方針にしている。
