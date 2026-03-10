# PR SUMMARY

## 変更点
- `train_prob_model.py` を leak-free な scaler fit と true walk-forward 評価に更新した。
- baseline を `static_baselines` と `online_baselines` に分離し、`eval_report_*.json` の後方互換を維持した。
- `manifest_*.json` を追加し、Streamlit で可視化できるようにした。
- `data_collector.py` に retry / timeout / 整合性チェック / 更新検知を追加した。
- `update_system.py` で収集 → 学習 → artifact 確認 → 推論 smoke test の導線を明確化した。
- `prediction_history_*.json` を追加し、holdout / walk-forward の各 draw ごとの予測上位番号と実当選番号の照合を保存するようにした。
- `app.py` に「✅ 実績との照合」タブを追加し、回別の hit 数サマリと一覧を確認できるようにした。
- `AGENT.md` と `docs/` を追加し、運用・評価・復旧手順を明文化した。

## 実装判断
- walk-forward fold は「初期 train 60% 以上、固定 test window=10、そこから最大 5 fold を等間隔に採用」とした。
- `online_baselines` は参考値として別表示にし、モデルの主比較は `static_baselines` に固定した。
- 追加依存は入れず、既存の TensorFlow / pandas / scikit-learn の範囲で実装した。
- `--preset` は無料 CPU 向けの導線として `default / fast / smoke` を用意し、明示指定がない限り現在の標準設定を使う。
- `--skip_final_train` は評価を先に回したいケース向けで、既存運用成果物が互換なら再利用し、無い場合だけ 0-epoch 雛形を保存して artifact 整合性を保つ。
- prediction history は評価用 artifact として保存し、今回の範囲では live 予測履歴とは分離した。将来の pending/resolved 方式追加時に互換を崩しにくくするため。
