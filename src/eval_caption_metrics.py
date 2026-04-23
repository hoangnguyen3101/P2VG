"""
Evaluate caption predictions from eval_caption.csv.

Computes:
- BLEU-1/2/3/4, ROUGE-1, ROUGE-L, METEOR, BERTScore
- Clinical Entity (CE) metrics via LLM semantic judging with an OpenAI-compatible API
"""
import argparse
import csv
import json
import os
import re
import time
from email.utils import parsedate_to_datetime

import numpy as np
import requests


CONDITION_SPECS = {
    "disc_herniation": {
        "description": "Focal disc herniation/protrusion/extrusion.",
    },
    "disc_bulge": {
        "description": "Broad-based disc bulge or bulging disc.",
    },
    "disc_degeneration": {
        "description": "Disc degeneration, desiccation, or reduced T2 signal.",
    },
    "annular_tear": {
        "description": "Annular tear or annular fissure.",
    },
    "disc_height_loss": {
        "description": "Disc height loss, reduced disc height, or collapsed disc space.",
    },
    "spinal_stenosis": {
        "description": "Central spinal canal stenosis or spinal stenosis.",
    },
    "foraminal_stenosis": {
        "description": "Neural foraminal narrowing or foraminal stenosis.",
    },
    "lateral_recess_stenosis": {
        "description": "Lateral recess stenosis or narrowing.",
    },
    "nerve_compression": {
        "description": "Nerve root compression, impingement, or contact.",
    },
    "spondylolisthesis": {
        "description": "Spondylolisthesis, anterolisthesis, or retrolisthesis.",
    },
    "compression_fracture": {
        "description": "Vertebral compression fracture or compression deformity.",
    },
    "osteophytes": {
        "description": "Degenerative osteophytes/spurs.",
    },
    "facet_arthropathy": {
        "description": "Facet arthropathy, facet hypertrophy, or facet joint degeneration.",
    },
    "curvature_abnormality": {
        "description": "Abnormal spinal curvature or loss/reversal of normal lumbar lordosis.",
    },
}


def compute_ce_metrics_from_labels(sample_labels, condition_names):
    """Compute CE precision/recall/F1 per condition and overall from boolean labels."""
    per_condition = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    for cond_name in condition_names:
        tp = fp = fn = tn = 0
        for labels in sample_labels:
            ref_has = bool(labels["reference"].get(cond_name, False))
            pred_has = bool(labels["prediction"].get(cond_name, False))

            if ref_has and pred_has:
                tp += 1
            elif not ref_has and pred_has:
                fp += 1
            elif ref_has and not pred_has:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn

        per_condition[cond_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    active = [v for v in per_condition.values() if v["support"] > 0]
    macro_p = np.mean([v["precision"] for v in active]) if active else 0.0
    macro_r = np.mean([v["recall"] for v in active]) if active else 0.0
    macro_f1 = np.mean([v["f1"] for v in active]) if active else 0.0

    return per_condition, {
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
    }
def build_llm_system_prompt(condition_specs):
    condition_lines = []
    for cond_name, spec in condition_specs.items():
        condition_lines.append("- {}: {}".format(cond_name, spec["description"]))

    return """You are an expert musculoskeletal radiology evaluator.

Your task is to compare a reference lumbar spine MRI report and a predicted report.
For each predefined clinical condition, determine whether the condition is positively present in:
1. the reference report
2. the prediction report

Rules:
- Use clinical meaning, not exact wording.
- Handle paraphrases and synonymous wording.
- Pay close attention to negation. If a report says a finding is absent, mark it false.
- Only mark true when the report positively states or clearly implies the finding.
- Do not infer findings that are not supported by the text.
- Return strict JSON only, no markdown and no extra commentary.

Conditions:
{}

Return exactly this JSON shape:
{{
  "conditions": {{
    "disc_herniation": {{
      "reference": false,
      "prediction": true,
      "reference_evidence": "",
      "prediction_evidence": "short quote or paraphrase"
    }}
  }}
}}""".format("\n".join(condition_lines))


def build_llm_system_prompt_fallback(condition_specs):
    condition_lines = []
    for cond_name, spec in condition_specs.items():
        condition_lines.append("- {}: {}".format(cond_name, spec["description"]))

    return """You are an expert musculoskeletal radiology evaluator.

Compare the reference lumbar spine MRI report and the predicted report.
For each predefined clinical condition, mark whether it is positively present in:
1. the reference report
2. the prediction report

Rules:
- Use clinical meaning, not exact wording.
- Respect negation carefully.
- Mark true only when the finding is explicitly stated or clearly implied.
- Do not add findings not supported by the report text.
- Output raw JSON only.
- Do not wrap the JSON in markdown fences.
- Include every condition exactly once.

Conditions:
{}

Required JSON format:
{{
  "conditions": {{
    "disc_herniation": {{
      "reference": false,
      "prediction": false,
      "reference_evidence": "",
      "prediction_evidence": ""
    }}
  }}
}}""".format("\n".join(condition_lines))


def build_llm_user_prompt(reference, prediction):
    return """Reference report:
{}

Prediction report:
{}""".format(reference.strip(), prediction.strip())


def build_response_schema(condition_specs):
    condition_properties = {}
    for cond_name in condition_specs:
        condition_properties[cond_name] = {
            "type": "object",
            "properties": {
                "reference": {"type": "boolean"},
                "prediction": {"type": "boolean"},
                "reference_evidence": {"type": "string"},
                "prediction_evidence": {"type": "string"},
            },
            "required": ["reference", "prediction", "reference_evidence", "prediction_evidence"],
            "additionalProperties": False,
        }

    return {
        "type": "object",
        "properties": {
            "conditions": {
                "type": "object",
                "properties": condition_properties,
                "required": list(condition_specs.keys()),
                "additionalProperties": False,
            }
        },
        "required": ["conditions"],
        "additionalProperties": False,
    }


def resolve_api_key(args):
    return (
        args.llm_api_key
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("LLM_API_KEY")
    )


def resolve_api_base(args):
    return (
        args.llm_api_base
        or os.getenv("GROQ_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("LLM_API_BASE")
        or "https://api.groq.com/openai/v1"
    ).rstrip("/")


def resolve_model(args):
    return (
        args.llm_model
        or os.getenv("GROQ_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("LLM_MODEL")
    )


def supports_structured_outputs(model):
    return model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}


def should_use_strict_schema(args, model):
    if args.llm_response_format == "json_schema":
        return True
    return args.llm_response_format == "auto" and supports_structured_outputs(model)


def parse_retry_after_seconds(value):
    if not value:
        return None

    value = str(value).strip()
    try:
        seconds = float(value)
        return max(0.0, seconds)
    except ValueError:
        pass

    try:
        retry_dt = parsedate_to_datetime(value)
        return max(0.0, retry_dt.timestamp() - time.time())
    except Exception:
        return None


def extract_json_object(text):
    text = text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    conditions_match = re.search(r'(\{\s*"conditions"\s*:\s*\{.*\}\s*\})', text, flags=re.DOTALL)
    if conditions_match:
        return json.loads(conditions_match.group(1))

    raise ValueError("LLM response does not contain valid JSON.")


def normalise_llm_labels(payload, condition_specs):
    conditions = payload.get("conditions", {})
    ref_labels = {}
    pred_labels = {}
    evidence = {}

    for cond_name in condition_specs:
        item = conditions.get(cond_name, {})
        ref_labels[cond_name] = bool(item.get("reference", False))
        pred_labels[cond_name] = bool(item.get("prediction", False))
        evidence[cond_name] = {
            "reference_evidence": str(item.get("reference_evidence", "") or ""),
            "prediction_evidence": str(item.get("prediction_evidence", "") or ""),
        }

    return {
        "reference": ref_labels,
        "prediction": pred_labels,
        "evidence": evidence,
    }


def call_openai_compatible_judge(reference, prediction, condition_specs, args):
    api_key = resolve_api_key(args)
    if not api_key:
        raise ValueError("Missing API key. Set --llm_api_key or GROQ_API_KEY.")

    api_base = resolve_api_base(args)
    model = resolve_model(args)
    if not model:
        raise ValueError("Missing model name. Set --llm_model or GROQ_MODEL.")

    url = api_base + "/chat/completions"
    payload = {
        "model": model,
        "temperature": args.llm_temperature,
        "messages": [
            {"role": "system", "content": build_llm_system_prompt(condition_specs)},
            {"role": "user", "content": build_llm_user_prompt(reference, prediction)},
        ],
    }
    if args.llm_response_format == "json_schema" or (
        args.llm_response_format == "auto" and supports_structured_outputs(model)
    ):
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "clinical_entity_judgment",
                "strict": should_use_strict_schema(args, model),
                "schema": build_response_schema(condition_specs),
            },
        }
    else:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
    }

    last_error = None
    added_json_repair_prompt = False
    for attempt in range(1, args.llm_max_retries + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=args.llm_timeout,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            parsed = extract_json_object(content)
            return normalise_llm_labels(parsed, condition_specs)
        except Exception as exc:
            last_error = exc
            if not isinstance(exc, requests.exceptions.HTTPError) and attempt < args.llm_max_retries:
                if not added_json_repair_prompt:
                    payload["messages"].append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response was not valid JSON. "
                                "Reply again with raw JSON only, matching the required schema exactly. "
                                "Do not include markdown, explanations, or extra text."
                            ),
                        }
                    )
                    added_json_repair_prompt = True
                print(
                    "Invalid JSON from provider. Retrying request {}/{}.".format(
                        attempt + 1, args.llm_max_retries
                    )
                )
            if attempt < args.llm_max_retries:
                delay = args.llm_retry_sleep
                if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
                    status_code = exc.response.status_code
                    if status_code == 429:
                        retry_after = parse_retry_after_seconds(exc.response.headers.get("Retry-After"))
                        if retry_after is not None:
                            delay = max(delay, retry_after)
                        else:
                            delay = max(delay, args.llm_rate_limit_backoff * attempt)
                        delay = min(delay, args.llm_max_backoff)
                        print(
                            "Rate limited by provider (429). Waiting {:.1f}s before retry {}/{}.".format(
                                delay, attempt + 1, args.llm_max_retries
                            )
                        )
                    elif status_code == 401:
                        raise RuntimeError(
                            "Unauthorized (401) from LLM provider. Check GROQ_API_KEY / --llm_api_key."
                        )
                    elif status_code == 400:
                        response_text = exc.response.text.strip()
                        if "json_validate_failed" in response_text:
                            if not added_json_repair_prompt:
                                payload["messages"].append(
                                    {
                                        "role": "user",
                                        "content": (
                                            "The provider rejected the previous response because it was not valid JSON. "
                                            "Reply again with raw JSON only, matching the required schema exactly. "
                                            "Do not include markdown, explanations, or extra text."
                                        ),
                                    }
                                )
                                added_json_repair_prompt = True
                            print("Provider rejected JSON output. Retrying with the same JSON response format.")
                            continue
                        raise RuntimeError(
                            "Bad Request (400) from LLM provider. Response body: {}".format(response_text[:4000])
                        )
                time.sleep(delay)

    raise RuntimeError("LLM judge failed after {} attempts: {}".format(args.llm_max_retries, last_error))


def load_existing_llm_details(details_path, condition_specs):
    if not details_path or not os.path.exists(details_path):
        return {}

    cached = {}
    with open(details_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            index = int(row["row_index"])
            ref_labels = {}
            pred_labels = {}
            evidence = {}
            for cond_name in condition_specs:
                ref_labels[cond_name] = row.get("{}__reference".format(cond_name), "False") == "True"
                pred_labels[cond_name] = row.get("{}__prediction".format(cond_name), "False") == "True"
                evidence[cond_name] = {
                    "reference_evidence": row.get("{}__reference_evidence".format(cond_name), ""),
                    "prediction_evidence": row.get("{}__prediction_evidence".format(cond_name), ""),
                }
            cached[index] = {
                "reference": ref_labels,
                "prediction": pred_labels,
                "evidence": evidence,
            }
    return cached


def write_llm_details(details_path, rows, condition_specs):
    fieldnames = ["row_index", "ground_truth", "prediction"]
    for cond_name in condition_specs:
        fieldnames.extend(
            [
                "{}__reference".format(cond_name),
                "{}__prediction".format(cond_name),
                "{}__reference_evidence".format(cond_name),
                "{}__prediction_evidence".format(cond_name),
            ]
        )

    with open(details_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compute_ce_metrics_llm(predictions, references, condition_specs, args):
    details_path = args.llm_details_csv
    cached = load_existing_llm_details(details_path, condition_specs) if args.llm_resume else {}

    sample_labels = []
    detail_rows = []
    total = len(predictions)

    for idx, (pred, ref) in enumerate(zip(predictions, references)):
        if idx in cached:
            labels = cached[idx]
        else:
            labels = call_openai_compatible_judge(ref, pred, condition_specs, args)
            if args.llm_request_interval > 0:
                time.sleep(args.llm_request_interval)

        sample_labels.append(labels)

        detail_row = {
            "row_index": idx,
            "ground_truth": ref,
            "prediction": pred,
        }
        for cond_name in condition_specs:
            detail_row["{}__reference".format(cond_name)] = labels["reference"][cond_name]
            detail_row["{}__prediction".format(cond_name)] = labels["prediction"][cond_name]
            detail_row["{}__reference_evidence".format(cond_name)] = labels["evidence"][cond_name]["reference_evidence"]
            detail_row["{}__prediction_evidence".format(cond_name)] = labels["evidence"][cond_name]["prediction_evidence"]
        detail_rows.append(detail_row)

        if details_path:
            write_llm_details(details_path, detail_rows, condition_specs)
        print("LLM judged sample {}/{}".format(idx + 1, total))

    metrics = compute_ce_metrics_from_labels(sample_labels, condition_specs.keys())
    return metrics, sample_labels


def print_ce_table(title, per_cond, overall):
    print("\n>>> {}".format(title))
    print("-" * 40)
    print("{:<25} {:>6} {:>6} {:>6} {:>4} {:>4} {:>4} {:>4}".format("Condition", "Prec", "Rec", "F1", "TP", "FP", "FN", "Sup"))
    print("-" * 70)
    for cond_name, metrics in per_cond.items():
        if metrics["support"] > 0 or metrics["fp"] > 0:
            print(
                "{:<25} {:>6.2f} {:>6.2f} {:>6.2f} {:>4} {:>4} {:>4} {:>4}".format(
                    cond_name,
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                    metrics["tp"],
                    metrics["fp"],
                    metrics["fn"],
                    metrics["support"],
                )
            )
    print("-" * 70)
    print("{:<25} {:>6.4f} {:>6.4f} {:>6.4f}".format("Micro Avg", overall["micro"]["precision"], overall["micro"]["recall"], overall["micro"]["f1"]))
    print("{:<25} {:>6.4f} {:>6.4f} {:>6.4f}".format("Macro Avg", overall["macro"]["precision"], overall["macro"]["recall"], overall["macro"]["f1"]))


def add_ce_scores(all_scores, prefix, per_cond, overall):
    for cond_name, metrics in per_cond.items():
        all_scores["{}_{}_P".format(prefix, cond_name)] = metrics["precision"]
        all_scores["{}_{}_R".format(prefix, cond_name)] = metrics["recall"]
        all_scores["{}_{}_F1".format(prefix, cond_name)] = metrics["f1"]
    all_scores["{}_Micro_P".format(prefix)] = overall["micro"]["precision"]
    all_scores["{}_Micro_R".format(prefix)] = overall["micro"]["recall"]
    all_scores["{}_Micro_F1".format(prefix)] = overall["micro"]["f1"]
    all_scores["{}_Macro_P".format(prefix)] = overall["macro"]["precision"]
    all_scores["{}_Macro_R".format(prefix)] = overall["macro"]["recall"]
    all_scores["{}_Macro_F1".format(prefix)] = overall["macro"]["f1"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate caption predictions")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="./output_gemma3/eval_results/eval_caption.csv",
        help="Path to eval_caption.csv from demo_csv.py",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to save eval scores CSV. Default: same dir as input, named eval_scores.csv",
    )
    parser.add_argument(
        "--llm_api_base",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL. Default: https://api.groq.com/openai/v1",
    )
    parser.add_argument(
        "--llm_api_key",
        type=str,
        default=None,
        help="API key for the LLM judge.",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default=None,
        help="Model name for the LLM judge.",
    )
    parser.add_argument(
        "--llm_response_format",
        type=str,
        default="auto",
        choices=["auto", "json_object", "json_schema"],
        help="Response formatting mode for the judge request.",
    )
    parser.add_argument(
        "--llm_schema_strict",
        action="store_true",
        help="Force strict JSON schema mode when --llm_response_format resolves to json_schema. Supported models in auto mode already use strict schema by default.",
    )
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    parser.add_argument("--llm_timeout", type=int, default=120)
    parser.add_argument("--llm_max_retries", type=int, default=8)
    parser.add_argument("--llm_retry_sleep", type=float, default=3.0)
    parser.add_argument(
        "--llm_rate_limit_backoff",
        type=float,
        default=10.0,
        help="Base backoff in seconds when the provider returns HTTP 429.",
    )
    parser.add_argument(
        "--llm_max_backoff",
        type=float,
        default=120.0,
        help="Maximum sleep time in seconds for one retry.",
    )
    parser.add_argument(
        "--llm_request_interval",
        type=float,
        default=12.0,
        help="Sleep time in seconds between successful judge requests.",
    )
    parser.add_argument(
        "--llm_details_csv",
        type=str,
        default=None,
        help="Path to save per-sample LLM CE judgments. Default: same dir as input, named eval_ce_llm_details.csv",
    )
    parser.add_argument(
        "--llm_resume",
        action="store_true",
        help="Reuse rows already saved in --llm_details_csv.",
    )
    parser.add_argument(
        "--skip_nlg",
        action="store_true",
        help="Skip BLEU/ROUGE/METEOR/BERTScore and compute only CE metrics.",
    )
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

    print("=" * 60)
    print("Loaded {} samples from {}".format(len(predictions), args.input_csv))
    print("=" * 60)

    if not args.skip_nlg:
        import evaluate

        print("\n>>> NLG Metrics")
        print("-" * 40)

        try:
            bleu = evaluate.load("bleu")
            bleu_overall = bleu.compute(
                predictions=predictions,
                references=[[r] for r in references],
            )
            all_scores["BLEU"] = bleu_overall["bleu"]
            print("BLEU:        {:.4f}".format(bleu_overall["bleu"]))
            for order in [1, 2, 3, 4]:
                bleu_result = bleu.compute(
                    predictions=predictions,
                    references=[[r] for r in references],
                    max_order=order,
                )
                all_scores["BLEU-{}".format(order)] = bleu_result["bleu"]
                print("BLEU-{}:      {:.4f}".format(order, bleu_result["bleu"]))
        except Exception as exc:
            print("BLEU error: {}".format(exc))

        try:
            rouge = evaluate.load("rouge")
            rouge_result = rouge.compute(
                predictions=predictions,
                references=references,
                rouge_types=["rouge1", "rougeL"],
            )
            all_scores["ROUGE-1"] = rouge_result["rouge1"]
            all_scores["ROUGE-L"] = rouge_result["rougeL"]
            print("ROUGE-1:     {:.4f}".format(rouge_result["rouge1"]))
            print("ROUGE-L:     {:.4f}".format(rouge_result["rougeL"]))
        except Exception as exc:
            print("ROUGE error: {}".format(exc))

        try:
            meteor = evaluate.load("meteor")
            meteor_result = meteor.compute(
                predictions=predictions,
                references=references,
            )
            all_scores["METEOR"] = meteor_result["meteor"]
            print("METEOR:      {:.4f}".format(meteor_result["meteor"]))
        except Exception as exc:
            print("METEOR error: {}".format(exc))

        try:
            bertscore = evaluate.load("bertscore")
            bert_result = bertscore.compute(
                predictions=predictions,
                references=references,
                lang="en",
            )
            avg_f1 = np.mean(bert_result["f1"])
            avg_precision = np.mean(bert_result["precision"])
            avg_recall = np.mean(bert_result["recall"])
            all_scores["BERTScore_P"] = avg_precision
            all_scores["BERTScore_R"] = avg_recall
            all_scores["BERTScore_F"] = avg_f1
            print("BERTScore P: {:.4f}".format(avg_precision))
            print("BERTScore R: {:.4f}".format(avg_recall))
            print("BERTScore F: {:.4f}".format(avg_f1))
        except Exception as exc:
            print("BERTScore error: {}".format(exc))

    (per_cond_llm, overall_llm), _ = compute_ce_metrics_llm(
        predictions,
        references,
        CONDITION_SPECS,
        args,
    )
    print_ce_table("Clinical Entity (CE) Metrics - LLM Judge", per_cond_llm, overall_llm)
    add_ce_scores(all_scores, "CE_LLM", per_cond_llm, overall_llm)

    with open(args.output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for key, value in all_scores.items():
            writer.writerow([key, "{:.4f}".format(value)])

    print("\nScores saved to: {}".format(args.output_csv))
    print("LLM CE details saved to: {}".format(args.llm_details_csv))
    print("=" * 60)


if __name__ == "__main__":
    main()
