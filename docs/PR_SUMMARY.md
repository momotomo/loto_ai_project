# PR SUMMARY

## 変更点
- `train_prob_model.py` を leak-free な scaler fit と true walk-forward 評価に更新した。
- baseline を `static_baselines` と `online_baselines` に分離し、`eval_report_*.json` の後方互換を維持した。
- `manifest_*.json` を追加し、Streamlit で可視化できるようにした。
- `data_collector.py` に retry / timeout / 整合性チェック / 更新検知を追加した。
- `update_system.py` で収集 → 学習 → artifact 確認 → 推論 smoke test の導線を明確化した。
- `prediction_history_*.json` を追加し、holdout / walk-forward の各 draw ごとの予測上位番号と実当選番号の照合を保存するようにした。
- `app.py` に「✅ 実績との照合」タブを追加し、回別の hit 数サマリと一覧を確認できるようにした。
- `app.py` に artifact 整合性チェックを追加し、`processed.csv` / `feature_cols.json` / `scaler.pkl` / `model.keras` の世代不一致を分かりやすく表示するようにした。
- Kaggle 同期 UI を `Kernel Ref (owner/kernel-slug)` に変更し、403 / 404 の切り分けメッセージと staged sync を追加した。
- Kaggle staged sync を global 判定から per-loto 判定へ変更し、今回対象外の loto_type は部分更新から安全にスキップできるようにした。
- `artifact_schema_version` / `bundle_id` を導入し、manifest / eval_report / prediction_history に bundle 世代情報を持たせた。
- per-loto clean sync で、更新対象 loto_type は新 bundle を final dir 上の temp file へ copy してから置換し、最後に不要な古いローカル artifact を掃除するようにした。
- Streamlit サイドバーに loto_type ごとのローカル artifact 削除ボタンを追加した。
- Kaggle 実行後の artifact を `/kaggle/working/app/...` から root の `/kaggle/working/data` / `/kaggle/working/models` へ export するようにし、Streamlit 同期は root 優先・`app/` fallback で読むようにした。
- Streamlit の staged sync は `/tmp` から final dir へ直接 rename せず、final dir 上の temp file へ copy してから replace する方式に変えた。cross-device failure で local bundle が空になるのを避けるため。
- `AGENT.md` と `docs/` を追加し、運用・評価・復旧手順を明文化した。

## 実装判断
- walk-forward fold は「初期 train 60% 以上、固定 test window=10、そこから最大 5 fold を等間隔に採用」とした。
- `online_baselines` は参考値として別表示にし、モデルの主比較は `static_baselines` に固定した。
- 追加依存は入れず、既存の TensorFlow / pandas / scikit-learn の範囲で実装した。
- `--preset` は無料 CPU 向けの導線として `default / fast / smoke` を用意し、明示指定がない限り現在の標準設定を使う。
- `--skip_final_train` は評価を先に回したいケース向けで、既存運用成果物が互換なら再利用し、無い場合だけ 0-epoch 雛形を保存して artifact 整合性を保つ。
- prediction history は評価用 artifact として保存し、今回の範囲では live 予測履歴とは分離した。将来の pending/resolved 方式追加時に互換を崩しにくくするため。
- 整合性エラー時は prediction tab のみ停止し、評価タブと実績照合タブは残す方針にした。artifact 世代の切り分けを UI 上で続けられるため。
- Kaggle 同期は download 後に bundle を検査し、不完全なら local replace を始めない方針にした。partial sync による世代混在を避けるため。
- ただし翌営業日の partial run では非対象 loto_type に `processed.csv` だけ残るので、bundle 検査も loto_type 単位に変更した。対象外はスキップ、対象だけ厳密更新とした。
- schema version 変更直後は古い local artifact との混在が起きやすいので、manifest の `artifact_schema_version` / `bundle_id` を見て clean sync する方針にした。初回移行時は全 loto_type を一度揃える運用を推奨する。
- Kaggle Output 側は path mismatch が起きやすいので、Output 直下の `data/` / `models/` を正とした。旧 Output との互換だけ Streamlit 側で吸収し、今後の運用は root export 前提に寄せる。
- Streamlit Cloud では `/tmp` と app 領域が別デバイスになりうるので、staged sync の最終反映で cross-device rename を前提にしない実装へ寄せた。artifact 不足はこの配置失敗の二次症状として扱う。
