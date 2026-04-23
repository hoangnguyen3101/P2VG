import pandas as pd
import os

BASE = "/home/hoangnv/AICD_HA/SPINE_BASE/P2VG"
folds = {
    "Fold 1": "output_gemma3",
    "Fold 2": "output_gemma3_fold2",
    "Fold 3": "output_gemma3_fold3",
    "Fold 4": "output_gemma3_fold4",
    "Fold 5": "output_gemma3_fold5",
}

all_dfs = []
print("=" * 65)
print(f"{'Fold':<10} {'Metric':<25} {'Value':>10}")
print("=" * 65)

for fold_name, folder in folds.items():
    path = os.path.join(BASE, folder, "eval_results", "eval_scores.csv")
    if not os.path.exists(path):
        print(f"{fold_name:<10} ⚠️  File not found: {path}")
        continue
    df = pd.read_csv(path)
    df.columns = ["Metric", "Value"]
    df["Fold"] = fold_name
    all_dfs.append(df)
    for _, row in df.iterrows():
        print(f"{fold_name:<10} {row['Metric']:<25} {row['Value']:>10.3f}")
    print("-" * 65)

if not all_dfs:
    print("No data found.")
else:
    combined = pd.concat(all_dfs, ignore_index=True)
    avg = combined.groupby("Metric")["Value"].agg(["mean", "std", "count"]).reset_index()
    avg.columns = ["Metric", "Mean", "Std", "N_Folds"]

    # Sắp xếp theo thứ tự ý nghĩa
    metric_order = ["BLEU", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
                    "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR", "BERTScore"]
    avg["_order"] = avg["Metric"].apply(
        lambda m: metric_order.index(m) if m in metric_order else 99
    )
    avg = avg.sort_values("_order").drop(columns="_order")

    print("\n" + "=" * 65)
    print(f"{'CROSS-FOLD AVERAGE':^65}")
    print("=" * 65)
    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'N_Folds':>8}")
    print("-" * 65)
    for _, row in avg.iterrows():
        print(f"{row['Metric']:<25} {row['Mean']:>10.3f} {row['Std']:>10.3f} {int(row['N_Folds']):>8}")

    # Lưu ra file — round 3 chữ số thập phân
    out_path = os.path.join(BASE, "eval_results_mean.csv")
    avg["Mean"] = avg["Mean"].round(3)
    avg["Std"] = avg["Std"].round(3)
    avg.to_csv(out_path, index=False)
    print(f"\n✅ Saved to: {out_path}")
