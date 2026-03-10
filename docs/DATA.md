# DATA

## データソース
- 取得元は `https://{loto_type}.thekyo.jp/data/{loto_type}.csv`。
- 対象は `miniloto` / `loto6` / `loto7`。

## 取得方法
- `requests.get(..., timeout=10)` を使用する。
- 最大 3 回の retry を行い、バックオフは 1 秒 / 2 秒 / 4 秒。
- 文字コードは `shift_jis` → `cp932` → `utf-8` の順で試す。

## 整合性チェック
- 必須カラム: `draw_id`, `date`, `num1..numN`。
- `draw_id` は数値化し、重複は最新行を残して除去、最終的に単調増加であることを確認する。
- 抽選数字は `1..max_num` に収まり、同一 draw 内で重複しないことを確認する。

## 更新検知
- 保存済み `data/*_raw.csv` の末尾 `draw_id` と比較し、更新あり / 更新なしをログに出す。
- 更新がなくても raw / processed を再保存し、後続処理の入力形式を揃える。

## 失敗時の挙動
- ネットワークや CSV 異常は `DataCollectionError` として扱う。
- CLI は失敗した loto_type を表示して non-zero exit する。
- `update_system.py` はその return code を見て後続処理を止める。
