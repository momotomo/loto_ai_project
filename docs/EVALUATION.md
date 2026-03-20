# EVALUATION

## walk-forward (rolling-origin)
- 評価対象は draw 単位の時系列サンプルで、各サンプルは直近 `LOOKBACK_WINDOW` 回を入力、次回 draw の multi-hot を正解とする。
- 入力表現は 2 系統ある。`legacy` は既存の tabular 特徴、`multihot` は 1 draw を `max_num` 次元 multi-hot とし、各番号に `hit / frequency / gap` を付与した時系列を使う。
- `legacy_holdout` は従来互換の単発 split。
- `walk_forward` は expanding-window 方式で train を広げ、固定長 test window を複数 fold で評価する。
- `legacy_holdout` と `walk_forward` はどちらも leak-free 評価で、scaler は train 期間だけで fit する。
- calibration は各 fold の train 末尾から切り出した calibration split だけで fit し、test には apply のみを行う。
- 一方で final model は運用モデル生成用なので、評価後に全データで fit し直して保存する。評価値と運用モデル生成は分離して考える。
- `smoke` preset は plumbing 検証用なので 0-epoch で動かし、配線・artifact・CLI/UI 互換の確認に使う。

## 指標
- LogLoss (BCE): draw × number の二値確率損失。
- Brier score: 確率予測の二乗誤差。
- Calibration: 予測確率を 10 bin に分けた平均予測確率と実測率。
- ECE: reliability bins から計算する expected calibration error。
- Top-k overlap: `k = pick_count` として、予測上位 k 個と実当選集合の重なり数ヒストグラム。

## post-hoc calibration
- `saved_calibration_method` は production 保存時に使う method を指定する。既定は `none`。
- `evaluation_calibration_methods` は評価時に比較する method 群で、既定は `none,temperature,isotonic`。
- `temperature` は依存が軽く、グローバルな確率スケール補正として扱いやすい。
- `isotonic` は単調変換で柔軟だが過学習しやすいため、calibration split を leak-free に切り出して比較専用に使う。
- fold ごとに raw と post-calibration の logloss / Brier / ECE を保存し、variant ごとに guardrail 付きで最終候補 method を選ぶ。

## 統計的比較
- `statistical_tests` では draw 単位の logloss 差 `candidate - reference` を保存する。負の値ほど candidate が良い。
- 各比較には bootstrap CI と paired permutation test を付与する。
- 比較時の予測列は各 variant の `selected_calibration_method` に揃える。raw 比較は `calibration_selection.raw_metrics` 側で残す。
- 最低限の比較対象:
  - `legacy_vs_best_static`
  - `multihot_vs_best_static`
  - `multihot_vs_legacy`
- 主比較は proper scoring rule である logloss を使う。Top-k は補助指標として解釈する。

## 採用判定ルール
- `decision_summary.rule` にルール文字列を保存する。
- 現行ルールは `multihot_vs_best_static` と `multihot_vs_legacy` の両方で、`bootstrap_ci.upper < 0` かつ `permutation_test.p_value < alpha` を満たし、さらに候補 calibration 後の logloss / Brier が raw 比で悪化せず、ECE が raw 改善または threshold 内にあるときだけ multihot を昇格候補とする。
- 条件を満たさない場合は既存 production variant を維持する。

## 集計評価と回別照合の違い
- `eval_report_*.json` は fold 単位と aggregate の集計指標を見るための artifact。
- `prediction_history_*.json` は各 draw の `predicted_top_k` と `actual_numbers` を照合し、「どの回で何個当たったか」を確認するための artifact。
- 集計で平均が良く見えても、回別に見ると hit が偏っていることがある。両方を合わせて解釈する。
- Streamlit では評価タブが集計、`✅ 実績との照合` タブが回別の比較を担当する。

## static baseline と online baseline
- `static_baselines`
  - `uniform`: 一様確率。
  - `frequency`: train 期間の出現頻度を固定で使う。
  - `gap`: train 末尾時点の gap 状態から作る固定予測。
- `online_baselines`
  - `frequency_online`: test 実測を逐次取り込んで頻度更新。
  - `gap_online`: test 実測で last-seen を逐次更新。
- モデルは test 中に再学習しないため、主比較は `static_baselines` に置く。

## リーク防止
- scaler は fold ごとに train 期間の観測済みデータだけで fit する。
- test 期間は transform のみを使う。
- `frequency` / `gap` など番号レベル特徴は prefix-causal に構築し、将来の draw を追加しても過去行の値が変わらないことを前提にする。
- calibration / overlap / baseline も fold ごとに算出し、最後に集計する。
- calibration fit に使う split は fold の train 末尾だけで、test draw の情報は使わない。
- prediction history も同じ test 区間の予測からだけ組み立て、future draw の情報は使わない。
- online baseline は test 実測を使うが、予測後にのみ状態更新する。

## preset の使い分け
- 通常: `venv/bin/python train_prob_model.py --loto_type loto6 --model_variant legacy --evaluation_model_variants legacy,multihot`
- calibration 比較つき: `venv/bin/python train_prob_model.py --loto_type loto6 --model_variant legacy --evaluation_model_variants legacy,multihot --saved_calibration_method none --evaluation_calibration_methods none,temperature,isotonic`
- 短時間確認: `venv/bin/python train_prob_model.py --loto_type loto6 --preset fast`
- 最小 smoke: `venv/bin/python train_prob_model.py --loto_type loto6 --preset smoke --model_variant legacy --evaluation_model_variants legacy,multihot`
- multihot 保存 smoke: `venv/bin/python train_prob_model.py --loto_type loto6 --preset smoke --model_variant multihot --evaluation_model_variants legacy,multihot --skip_final_train`
- 評価だけ更新したい場合: `venv/bin/python train_prob_model.py --loto_type loto6 --preset smoke --skip_final_train`
- 実験追跡込み: `venv/bin/python scripts/run_experiment.py --config-json '{"loto_type":"loto6","preset":"smoke","seed":42,"model_variant":"multihot","evaluation_model_variants":"legacy,multihot","refresh_data":false,"skip_final_train":true}'`

## 再現性メモ
- `eval_report_*.json` と `manifest_*.json` には `data_fingerprint` / `training_context` / `runtime_environment` を保存する。
- これにより data hash、preprocessing version、preset、seed、model variant、主要 hyperparameter、Python / dependency versions を artifact 単位で追える。
