import os
import json
from config import LOTO_CONFIG

DATA_DIR = "data"

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" 🕵️ 宝くじ AI Walk-Forward 評価レポート (リークなし)")
    print("="*70 + "\n")
    
    for ltype, config in LOTO_CONFIG.items():
        report_path = os.path.join(DATA_DIR, f"eval_report_{ltype}.json")
        if not os.path.exists(report_path):
            print(f"[{config['name']}] の評価レポートが見つかりません。")
            continue
            
        with open(report_path, "r") as f:
            report = json.load(f)
            
        print(f"📊 【{config['name']}】 テスト対象: {report['test_samples']} 件")
        print(f"{'モデル/ベースライン':<20} | {'LogLoss':<10} | {'Brier':<10} | {'Top-K重なり平均':<15}")
        print("-" * 65)
        
        # モデルの結果
        m = report["Model (LSTM)"]
        print(f"{'★ AI Model (LSTM)':<20} | {m['logloss']:<10.4f} | {m['brier']:<10.4f} | {m['mean_overlap_top_k']:<15.3f}")
        
        # ベースラインの結果
        for b_name, b_metrics in report["Baselines"].items():
            print(f"{b_name:<20} | {b_metrics['logloss']:<10.4f} | {b_metrics['brier']:<10.4f} | {b_metrics['mean_overlap_top_k']:<15.3f}")
            
        # 当選分布
        dist = m['overlap_dist']
        dist_str = ", ".join(f"{k}個:{v}回" for k, v in dist.items())
        print(f"\n   [AI Model Top-{config['pick_count']} の的中数分布] -> {dist_str}\n")