"""
Evaluate caption predictions from eval_caption.csv
Computes: BLEU-1, ROUGE-1, ROUGE-L, METEOR, BERTScore F1, CE metrics
"""
import csv
import argparse
import numpy as np

# Clinical Entity conditions for spine MRI
CE_CONDITIONS = {
    # Disc pathology
    "disc_herniation": ["disc herniation", "disc protrusion", "disc extrusion"],
    "disc_bulge": ["disc bulge", "bulging disc"],
    "disc_degeneration": ["degeneration", "degenerative", "signal reduction"],
    "annular_tear": ["annular tear", "annular fissure"],
    "disc_height_loss": ["disc height", "height loss", "height reduction"],
    # Stenosis
    "spinal_stenosis": ["spinal stenosis", "spinal canal stenosis", "canal stenosis"],
    "foraminal_stenosis": ["foraminal stenosis", "neural foraminal"],
    "lateral_recess_stenosis": ["lateral recess stenosis", "lateral recess narrowing"],
    # Nerve
    "nerve_compression": ["nerve root compression", "nerve root contact", "nerve root impingement"],
    # Bone
    "spondylolisthesis": ["spondylolisthesis", "anterolisthesis", "retrolisthesis"],
    "compression_fracture": ["compression fracture", "vertebral fracture"],
    "osteophytes": ["osteophyte", "osteophytes"],
    # Other
    "facet_arthropathy": ["facet joint", "facet hypertrophy", "facet arthropathy"],
    "curvature_abnormality": ["scoliosis", "kyphosis", "lordosis"],
}


def detect_conditions(text, conditions):
    """Check which conditions are present in text (case-insensitive)."""
    text_lower = text.lower()
    detected = {}
    for cond_name, keywords in conditions.items():
        detected[cond_name] = any(kw.lower() in text_lower for kw in keywords)
    return detected


def compute_ce_metrics(predictions, references, conditions):
    """Compute Clinical Entity precision, recall, F1 per condition and overall."""
    per_condition = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    for cond_name in conditions:
        tp = fp = fn = tn = 0
        for pred, ref in zip(predictions, references):
            pred_labels = detect_conditions(pred, conditions)
            ref_labels = detect_conditions(ref, conditions)
            if ref_labels[cond_name] and pred_labels[cond_name]:
                tp += 1
            elif not ref_labels[cond_name] and pred_labels[cond_name]:
                fp += 1
            elif ref_labels[cond_name] and not pred_labels[cond_name]:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn  # number of GT positives

        per_condition[cond_name] = {
            "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "support": support
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Micro average
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    # Macro average (only conditions with support > 0)
    active = [v for v in per_condition.values() if v["support"] > 0]
    macro_p = np.mean([v["precision"] for v in active]) if active else 0.0
    macro_r = np.mean([v["recall"] for v in active]) if active else 0.0
    macro_f1 = np.mean([v["f1"] for v in active]) if active else 0.0

    return per_condition, {
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate caption predictions")
    parser.add_argument("--input_csv", type=str, 
                        default="./output_spine_dual_v1/eval_results/eval_caption.csv",
                        help="Path to eval_caption.csv from demo_csv.py")
    parser.add_argument("--output_csv", type=str,
                        default=None,
                        help="Path to save eval scores CSV. Default: same dir as input, named eval_scores.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_csv is None:
        import os
        args.output_csv = os.path.join(os.path.dirname(args.input_csv), "eval_scores.csv")

    all_scores = {}  # collect all metrics for CSV

    # Load predictions and ground truths
    predictions = []
    references = []
    with open(args.input_csv, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt = row["Ground Truth"].strip()
            pred = row["pred"].strip()
            if gt and pred:
                predictions.append(pred)
                references.append(gt)

    print("=" * 60)
    print(f"Loaded {len(predictions)} samples from {args.input_csv}")
    print("=" * 60)

    # ================ NLG Metrics ================
    print("\n>>> NLG Metrics")
    print("-" * 40)

    # --- BLEU ---
    try:
        import evaluate
        bleu = evaluate.load("bleu")
        # Overall BLEU (default: geometric mean of 1~4 gram with brevity penalty)
        bleu_overall = bleu.compute(
            predictions=predictions,
            references=[[r] for r in references],
        )
        all_scores["BLEU"] = bleu_overall['bleu']
        print(f"BLEU:        {bleu_overall['bleu']:.4f}")
        # Per n-gram
        for order in [1, 2, 3, 4]:
            bleu_result = bleu.compute(
                predictions=predictions, 
                references=[[r] for r in references], 
                max_order=order
            )
            all_scores[f"BLEU-{order}"] = bleu_result['bleu']
            print(f"BLEU-{order}:      {bleu_result['bleu']:.4f}")
    except Exception as e:
        print(f"BLEU error: {e}")

    # --- ROUGE ---
    try:
        rouge = evaluate.load("rouge")
        rouge_result = rouge.compute(
            predictions=predictions, 
            references=references, 
            rouge_types=["rouge1", "rougeL"]
        )
        all_scores["ROUGE-1"] = rouge_result['rouge1']
        all_scores["ROUGE-L"] = rouge_result['rougeL']
        print(f"ROUGE-1:     {rouge_result['rouge1']:.4f}")
        print(f"ROUGE-L:     {rouge_result['rougeL']:.4f}")
    except Exception as e:
        print(f"ROUGE error: {e}")

    # --- METEOR ---
    try:
        meteor = evaluate.load("meteor")
        meteor_result = meteor.compute(
            predictions=predictions, 
            references=references
        )
        all_scores["METEOR"] = meteor_result['meteor']
        print(f"METEOR:      {meteor_result['meteor']:.4f}")
    except Exception as e:
        print(f"METEOR error: {e}")

    # --- BERTScore ---
    try:
        bertscore = evaluate.load("bertscore")
        bert_result = bertscore.compute(
            predictions=predictions, 
            references=references, 
            lang="en"
        )
        avg_f1 = np.mean(bert_result["f1"])
        avg_precision = np.mean(bert_result["precision"])
        avg_recall = np.mean(bert_result["recall"])
        all_scores["BERTScore_P"] = avg_precision
        all_scores["BERTScore_R"] = avg_recall
        all_scores["BERTScore_F"] = avg_f1
        print(f"BERTScore P: {avg_precision:.4f}")
        print(f"BERTScore R: {avg_recall:.4f}")
        print(f"BERTScore F: {avg_f1:.4f}")
    except Exception as e:
        print(f"BERTScore error: {e}")

    # ================ CE Metrics ================
    print("\n>>> Clinical Entity (CE) Metrics")
    print("-" * 40)

    per_cond, overall = compute_ce_metrics(predictions, references, CE_CONDITIONS)

    # Per-condition table
    print(f"{'Condition':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'Sup':>4}")
    print("-" * 70)
    for cond_name, m in per_cond.items():
        if m["support"] > 0 or m["fp"] > 0:
            print(f"{cond_name:<25} {m['precision']:>6.2f} {m['recall']:>6.2f} {m['f1']:>6.2f} "
                  f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['support']:>4}")
    print("-" * 70)

    # Overall
    print(f"{'Micro Avg':<25} {overall['micro']['precision']:>6.4f} {overall['micro']['recall']:>6.4f} {overall['micro']['f1']:>6.4f}")
    print(f"{'Macro Avg':<25} {overall['macro']['precision']:>6.4f} {overall['macro']['recall']:>6.4f} {overall['macro']['f1']:>6.4f}")

    # Add CE scores to all_scores
    for cond_name, m in per_cond.items():
        all_scores[f"CE_{cond_name}_P"] = m["precision"]
        all_scores[f"CE_{cond_name}_R"] = m["recall"]
        all_scores[f"CE_{cond_name}_F1"] = m["f1"]
    all_scores["CE_Micro_P"] = overall["micro"]["precision"]
    all_scores["CE_Micro_R"] = overall["micro"]["recall"]
    all_scores["CE_Micro_F1"] = overall["micro"]["f1"]
    all_scores["CE_Macro_P"] = overall["macro"]["precision"]
    all_scores["CE_Macro_R"] = overall["macro"]["recall"]
    all_scores["CE_Macro_F1"] = overall["macro"]["f1"]

    # ================ Save to CSV ================
    import os
    with open(args.output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in all_scores.items():
            writer.writerow([k, f"{v:.4f}"])
    print(f"\nScores saved to: {args.output_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
