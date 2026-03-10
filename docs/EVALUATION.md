# EVALUATION

## walk-forward (rolling-origin)
- 評価対象は draw 単位の時系列サンプルで、各サンプルは直近 `LOOKBACK_WINDOW` 回を入力、次回 draw の multi-hot を正解とする。
- `legacy_holdout` は従来互換の単発 split。
- `walk_forward` は expanding-window 方式で train を広げ、固定長 test window を複数 fold で評価する。
- `legacy_holdout` と `walk_forward` はどちらも leak-free 評価で、scaler は train 期間だけで fit する。
- 一方で final model は運用モデル生成用なので、評価後に全データで fit し直して保存する。評価値と運用モデル生成は分離して考える。

## 指標
- LogLoss (BCE): draw × number の二値確率損失。
- Brier score: 確率予測の二乗誤差。
- Calibration: 予測確率を 10 bin に分けた平均予測確率と実測率。
- Top-k overlap: `k = pick_count` として、予測上位 k 個と実当選集合の重なり数ヒストグラム。

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
- calibration / overlap / baseline も fold ごとに算出し、最後に集計する。
- prediction history も同じ test 区間の予測からだけ組み立て、future draw の情報は使わない。
- online baseline は test 実測を使うが、予測後にのみ状態更新する。

## preset の使い分け
- 通常: `venv/bin/python train_prob_model.py --loto_type loto6`
- 短時間確認: `venv/bin/python train_prob_model.py --loto_type loto6 --preset fast`
- 最小 smoke: `venv/bin/python train_prob_model.py --loto_type loto6 --preset smoke`
- 評価だけ更新したい場合: `venv/bin/python train_prob_model.py --loto_type loto6 --preset smoke --skip_final_train`
