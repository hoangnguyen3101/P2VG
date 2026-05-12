"""Evaluate caption predictions from eval_caption.csv (BLEU, ROUGE, METEOR, BERTScore, CE)."""
import argparse
import csv
import os

import numpy as np
from loguru import logger

from p2vg.eval.metrics import (
    CONDITION_SPECS,
    add_ce_scores,
    compute_ce_metrics_llm,
    print_ce_table,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate caption predictions")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="./outputs/eval_results/eval_caption.csv",
        help="Path to eval_caption.csv from scripts/demo_csv.py",
    )
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--llm_api_base", type=str, default=None)
    parser.add_argument("--llm_api_key", type=str, default=None)
    parser.add_argument("--llm_model", type=str, default=None)
    parser.add_argument(
        "--llm_response_format",
        type=str,
        default="auto",
        choices=["auto", "json_object", "json_schema"],
    )
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    parser.add_argument("--llm_timeout", type=int, default=120)
    parser.add_argument("--llm_max_retries", type=int, default=8)
    parser.add_argument("--llm_retry_sleep", type=float, default=3.0)
    parser.add_argument("--llm_rate_limit_backoff", type=float, default=10.0)
    parser.add_argument("--llm_max_backoff", type=float, default=120.0)
    parser.add_argument("--llm_request_interval", type=float, default=12.0)
    parser.add_argument("--llm_details_csv", type=str, default=None)
    parser.add_argument("--llm_resume", action="store_true")
    parser.add_argument("--skip_nlg", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_csv is None:
        args.output_csv = os.path.join(os.path.dirname(args.input_csv), "eval_scores.csv")
    if args.llm_details_csv is None:
        args.llm_details_csv = os.path.join(os.path.dirname(args.input_csv), "eval_ce_llm_details.csv")

    all_scores = {}
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

    logger.info("Loaded {} samples from {}", len(predictions), args.input_csv)

    if not args.skip_nlg:
        import evaluate

        logger.info("Computing NLG metrics...")
        try:
            bleu = evaluate.load("bleu")
            for order in [1, 2, 3, 4]:
                result = bleu.compute(
                    predictions=predictions,
                    references=[[r] for r in references],
                    max_order=order,
                )
                all_scores[f"BLEU-{order}"] = result["bleu"]
                logger.info("BLEU-{}: {:.4f}", order, result["bleu"])
        except Exception as exc:
            logger.warning("BLEU error: {}", exc)

        try:
            rouge = evaluate.load("rouge")
            result = rouge.compute(
                predictions=predictions,
                references=references,
                rouge_types=["rouge1", "rougeL"],
            )
            all_scores["ROUGE-1"] = result["rouge1"]
            all_scores["ROUGE-L"] = result["rougeL"]
            logger.info("ROUGE-1: {:.4f}  ROUGE-L: {:.4f}", result["rouge1"], result["rougeL"])
        except Exception as exc:
            logger.warning("ROUGE error: {}", exc)

        try:
            meteor = evaluate.load("meteor")
            result = meteor.compute(predictions=predictions, references=references)
            all_scores["METEOR"] = result["meteor"]
            logger.info("METEOR: {:.4f}", result["meteor"])
        except Exception as exc:
            logger.warning("METEOR error: {}", exc)

        try:
            bertscore = evaluate.load("bertscore")
            result = bertscore.compute(predictions=predictions, references=references, lang="en")
            all_scores["BERTScore_P"] = float(np.mean(result["precision"]))
            all_scores["BERTScore_R"] = float(np.mean(result["recall"]))
            all_scores["BERTScore_F"] = float(np.mean(result["f1"]))
            logger.info(
                "BERTScore P={:.4f} R={:.4f} F={:.4f}",
                all_scores["BERTScore_P"],
                all_scores["BERTScore_R"],
                all_scores["BERTScore_F"],
            )
        except Exception as exc:
            logger.warning("BERTScore error: {}", exc)

    (per_cond_llm, overall_llm), _ = compute_ce_metrics_llm(
        predictions, references, CONDITION_SPECS, args
    )
    print_ce_table("Clinical Entity (CE) Metrics - LLM Judge", per_cond_llm, overall_llm)
    add_ce_scores(all_scores, "CE_LLM", per_cond_llm, overall_llm)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for key, value in all_scores.items():
            writer.writerow([key, "{:.4f}".format(value)])

    logger.info("Scores saved to: {}", args.output_csv)
    logger.info("LLM CE details saved to: {}", args.llm_details_csv)


if __name__ == "__main__":
    main()
