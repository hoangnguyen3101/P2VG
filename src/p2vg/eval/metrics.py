"""
Caption evaluation metrics: CE (Clinical Entity) and NLG (BLEU, ROUGE, METEOR, BERTScore).
"""
import csv
import json
import os
import re
import time
from email.utils import parsedate_to_datetime

import numpy as np
import requests
from loguru import logger


CONDITION_SPECS = {
    "disc_herniation": {"description": "Focal disc herniation/protrusion/extrusion."},
    "disc_bulge": {"description": "Broad-based disc bulge or bulging disc."},
    "disc_degeneration": {"description": "Disc degeneration, desiccation, or reduced T2 signal."},
    "spinal_stenosis": {"description": "Central spinal canal stenosis or spinal stenosis."},
    "osteophytes": {"description": "Degenerative osteophytes/spurs."},
    "nerve_compression": {"description": "Nerve root compression, impingement, or contact."},
    "curvature_abnormality": {"description": "Abnormal spinal curvature or loss/reversal of normal lumbar lordosis."},
    "annular_tear": {"description": "Annular tear or annular fissure."},
    "disc_height_loss": {"description": "Disc height loss, reduced disc height, or collapsed disc space."},
}


def compute_ce_metrics_from_labels(sample_labels, condition_names):
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

        per_condition[cond_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": tp + fn,
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
    condition_lines = [
        "- {}: {}".format(name, spec["description"])
        for name, spec in condition_specs.items()
    ]
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


def build_llm_user_prompt(reference, prediction):
    return "Reference report:\n{}\n\nPrediction report:\n{}".format(
        reference.strip(), prediction.strip()
    )


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
        getattr(args, "llm_api_key", None)
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("LLM_API_KEY")
    )


def resolve_api_base(args):
    return (
        getattr(args, "llm_api_base", None)
        or os.getenv("GROQ_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("LLM_API_BASE")
        or "https://api.groq.com/openai/v1"
    ).rstrip("/")


def resolve_model(args):
    return (
        getattr(args, "llm_model", None)
        or os.getenv("GROQ_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("LLM_MODEL")
    )


def supports_structured_outputs(model):
    return model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}


def parse_retry_after_seconds(value):
    if not value:
        return None
    value = str(value).strip()
    try:
        return max(0.0, float(value))
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
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError("LLM response does not contain valid JSON.")


def normalise_llm_labels(payload, condition_specs):
    conditions = payload.get("conditions", {})
    ref_labels, pred_labels, evidence = {}, {}, {}
    for cond_name in condition_specs:
        item = conditions.get(cond_name, {})
        ref_labels[cond_name] = bool(item.get("reference", False))
        pred_labels[cond_name] = bool(item.get("prediction", False))
        evidence[cond_name] = {
            "reference_evidence": str(item.get("reference_evidence", "") or ""),
            "prediction_evidence": str(item.get("prediction_evidence", "") or ""),
        }
    return {"reference": ref_labels, "prediction": pred_labels, "evidence": evidence}


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
        "temperature": getattr(args, "llm_temperature", 0.0),
        "messages": [
            {"role": "system", "content": build_llm_system_prompt(condition_specs)},
            {"role": "user", "content": build_llm_user_prompt(reference, prediction)},
        ],
    }

    response_format = getattr(args, "llm_response_format", "auto")
    if response_format == "json_schema" or (
        response_format == "auto" and supports_structured_outputs(model)
    ):
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "clinical_entity_judgment",
                "strict": True,
                "schema": build_response_schema(condition_specs),
            },
        }
    else:
        payload["response_format"] = {"type": "json_object"}

    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    max_retries = getattr(args, "llm_max_retries", 8)
    retry_sleep = getattr(args, "llm_retry_sleep", 3.0)
    rate_limit_backoff = getattr(args, "llm_rate_limit_backoff", 10.0)
    max_backoff = getattr(args, "llm_max_backoff", 120.0)
    timeout = getattr(args, "llm_timeout", 120)

    last_error = None
    added_json_repair_prompt = False
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            parsed = extract_json_object(content)
            return normalise_llm_labels(parsed, condition_specs)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                delay = retry_sleep
                if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
                    status_code = exc.response.status_code
                    if status_code == 429:
                        retry_after = parse_retry_after_seconds(exc.response.headers.get("Retry-After"))
                        if retry_after is not None:
                            delay = max(delay, retry_after)
                        else:
                            delay = max(delay, rate_limit_backoff * attempt)
                        delay = min(delay, max_backoff)
                    elif status_code == 401:
                        raise RuntimeError("Unauthorized (401) from LLM provider.")
                    elif status_code == 400:
                        if not added_json_repair_prompt:
                            payload["messages"].append({
                                "role": "user",
                                "content": (
                                    "Your previous response was not valid JSON. "
                                    "Reply again with raw JSON only, matching the required schema exactly."
                                ),
                            })
                            added_json_repair_prompt = True
                        continue
                if not added_json_repair_prompt and not isinstance(exc, requests.exceptions.HTTPError):
                    payload["messages"].append({
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON. "
                            "Reply again with raw JSON only, matching the required schema exactly."
                        ),
                    })
                    added_json_repair_prompt = True
                time.sleep(delay)

    raise RuntimeError("LLM judge failed after {} attempts: {}".format(max_retries, last_error))


def load_existing_llm_details(details_path, condition_specs):
    if not details_path or not os.path.exists(details_path):
        return {}
    cached = {}
    with open(details_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            index = int(row["row_index"])
            ref_labels, pred_labels, evidence = {}, {}, {}
            for cond_name in condition_specs:
                ref_labels[cond_name] = row.get("{}__reference".format(cond_name), "False") == "True"
                pred_labels[cond_name] = row.get("{}__prediction".format(cond_name), "False") == "True"
                evidence[cond_name] = {
                    "reference_evidence": row.get("{}__reference_evidence".format(cond_name), ""),
                    "prediction_evidence": row.get("{}__prediction_evidence".format(cond_name), ""),
                }
            cached[index] = {"reference": ref_labels, "prediction": pred_labels, "evidence": evidence}
    return cached


def write_llm_details(details_path, rows, condition_specs):
    fieldnames = ["row_index", "ground_truth", "prediction"]
    for cond_name in condition_specs:
        fieldnames.extend([
            "{}__reference".format(cond_name),
            "{}__prediction".format(cond_name),
            "{}__reference_evidence".format(cond_name),
            "{}__prediction_evidence".format(cond_name),
        ])
    with open(details_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compute_ce_metrics_llm(predictions, references, condition_specs, args):
    details_path = getattr(args, "llm_details_csv", None)
    llm_resume = getattr(args, "llm_resume", False)
    cached = load_existing_llm_details(details_path, condition_specs) if llm_resume else {}

    sample_labels = []
    detail_rows = []
    total = len(predictions)

    for idx, (pred, ref) in enumerate(zip(predictions, references)):
        if idx in cached:
            labels = cached[idx]
        else:
            labels = call_openai_compatible_judge(ref, pred, condition_specs, args)
            request_interval = getattr(args, "llm_request_interval", 0.0)
            if request_interval > 0:
                time.sleep(request_interval)

        sample_labels.append(labels)

        detail_row = {"row_index": idx, "ground_truth": ref, "prediction": pred}
        for cond_name in condition_specs:
            detail_row["{}__reference".format(cond_name)] = labels["reference"][cond_name]
            detail_row["{}__prediction".format(cond_name)] = labels["prediction"][cond_name]
            detail_row["{}__reference_evidence".format(cond_name)] = labels["evidence"][cond_name]["reference_evidence"]
            detail_row["{}__prediction_evidence".format(cond_name)] = labels["evidence"][cond_name]["prediction_evidence"]
        detail_rows.append(detail_row)

        if details_path:
            write_llm_details(details_path, detail_rows, condition_specs)
        logger.info("LLM judged sample {}/{}", idx + 1, total)

    metrics = compute_ce_metrics_from_labels(sample_labels, condition_specs.keys())
    return metrics, sample_labels


def print_ce_table(title, per_cond, overall):
    logger.info("\n>>> {}", title)
    logger.info("-" * 40)
    for cond_name, metrics in per_cond.items():
        if metrics["support"] > 0 or metrics["fp"] > 0:
            logger.info(
                "{:<25} P={:.2f} R={:.2f} F1={:.2f} TP={} FP={} FN={} Sup={}",
                cond_name,
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                metrics["tp"],
                metrics["fp"],
                metrics["fn"],
                metrics["support"],
            )
    logger.info("-" * 70)
    logger.info("Micro Avg  P={:.4f} R={:.4f} F1={:.4f}", overall["micro"]["precision"], overall["micro"]["recall"], overall["micro"]["f1"])
    logger.info("Macro Avg  P={:.4f} R={:.4f} F1={:.4f}", overall["macro"]["precision"], overall["macro"]["recall"], overall["macro"]["f1"])


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
