#!/usr/bin/env python3
"""
test_set_pipeline.py

Test-set friendly pipeline to:
1) load patients CSV
2) map evidence codes to symptoms/answers (code-level + human-readable)
3) build nested per-patient payloads:
     {
       "demographics": {"AGE": <int/str>, "SEX": <str>},
       "evidence": { "<question>": <answer>, ... }
     }
4) provide a placeholder to call an LLM with these payloads
5) accept LLM results (diagnosis + probs) via function or --predictions
6) compare with ground truth using simple metrics

Usage (example):
  python test_set_pipeline.py \
      --patients data/test_patients.csv \
      --evidences data/evidences.json \
      --conditions data/conditions.json \
      --label-col GROUND_TRUTH \
      --limit 200 \
      --out-dir outputs

Predictions file format (JSONL), one object per patient (index order = CSV after any filtering):
  {"predicted_label": "Influenza", "probs": {"Influenza": 0.72, "URTI": 0.20, "Pneumonia": 0.08}}
"""

import argparse
import ast
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# --------------------------
# Loading utilities
# --------------------------

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_patients_csv(path: str, limit: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if limit is not None:
        df = df.head(limit).copy()
    return df


# --------------------------
# Evidence parsing & mapping
# --------------------------

def parse_evidence_cell(cell: Any) -> Dict[str, Any]:
    """
    Parse a single EVIDENCES cell (e.g., "['E_91','E_55_@_V_123']" or "E_91,E_55_@_V_123").
    Returns dict like {"E_91": True, "E_55": "V_123"}.
    """
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return {}
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return {}
        try:
            items = ast.literal_eval(s)
            if not isinstance(items, (list, tuple)):
                items = [x.strip() for x in s.split(",") if x.strip()]
        except Exception:
            items = [x.strip() for x in s.split(",") if x.strip()]
    elif isinstance(cell, (list, tuple)):
        items = list(cell)
    else:
        items = [str(cell)]

    out: Dict[str, Any] = {}
    for e in items:
        if not e:
            continue
        if "_@_" in e:
            code, val = e.split("_@_", 1)
        else:
            code, val = e, True
        out[code.strip()] = val
    return out


def make_human_readable(
    evidence_codes: Dict[str, Any],
    evidences_meta: Dict[str, Any],
    language: str = "en",
) -> Dict[str, Any]:
    """
    Convert code-level dict (e.g., {'E_55': 'V_123', 'E_91': True})
    into a human-readable dict using evidences.json:
      - Replace E_* with question text (question_en/question_fr)
      - Map V_* values via 'value_meaning' when available
    """
    readable: Dict[str, Any] = {}
    for e_code, val in evidence_codes.items():
        meta = evidences_meta.get(e_code) or evidences_meta.get(e_code.strip())
        if not meta:
            # Unknown code: keep as-is under code key
            readable[e_code] = val
            continue

        q_key = "question_en" if language.lower().startswith("en") else "question_fr"
        question = meta.get(q_key) or meta.get("question_en") or meta.get("name") or e_code

        if isinstance(val, str) and val.startswith("V_"):
            vm = meta.get("value_meaning", {})
            vinfo = vm.get(val)
            if isinstance(vinfo, dict):
                mapped = vinfo.get("en") or vinfo.get("fr") or val
            else:
                mapped = vinfo or val
            readable[question] = mapped
        else:
            readable[question] = val
    return readable


def build_patient_evidence_dicts(
    df: pd.DataFrame,
    evidences_meta: Dict[str, Any],
    allowed_codes: Optional[set] = None,
    language: str = "en",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build two parallel lists:
      - code_dicts: per-patient code-level dicts
      - human_dicts: per-patient human-readable dicts
    """
    if "EVIDENCES" not in df.columns:
        raise ValueError("CSV must include an 'EVIDENCES' column.")

    code_dicts: List[Dict[str, Any]] = []
    human_dicts: List[Dict[str, Any]] = []

    for cell in df["EVIDENCES"]:
        codes = parse_evidence_cell(cell)
        if allowed_codes is not None:
            codes = {k: v for k, v in codes.items() if k in allowed_codes}

        code_dicts.append(codes)
        human_dicts.append(make_human_readable(codes, evidences_meta, language=language))

    return code_dicts, human_dicts


# --------------------------
# Preview printing
# --------------------------

def print_preview_dicts(dicts: List[Dict[str, Any]], n: int = 100, title: str = "objects") -> None:
    print(f"=== Preview of first {min(n, len(dicts))} {title} ===")
    for i, d in enumerate(dicts[:n]):
        print(f"Patient {i}:\n{json.dumps(d, indent=2, ensure_ascii=False)}")
        print("-" * 60)


# --------------------------
# LLM placeholder
# --------------------------

def run_llm_pipeline(patient_payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run MYCIN inference engine on patient payloads.
    
    Uses LLM only for answering questions, not for rule evaluation.
    Rule evaluation is handled programmatically by the inference engine.

    Input: list of payloads, each like:
      {
        "demographics": {"AGE": 49, "SEX": "F"},
        "evidence": {"<question>": <answer>, ...}
      }

    Expected output (aligned to input order), each element like:
      {
        "predicted_label": "streptococcus",
        "probs": {"streptococcus": 0.72, "staphylococcus": 0.20, ...}
      }
    """
    try:
        from mycin_pipeline_integration import run_mycin_pipeline, example_llm_call
    except ImportError:
        raise NotImplementedError(
            "MYCIN modules not found. Install dependencies or use --predictions JSONL."
        )
    
    # Use MYCIN pipeline
    # Replace example_llm_call with your actual LLM API call function
    return run_mycin_pipeline(
        patient_payloads,
        llm_call_fn=example_llm_call,  # Replace with your LLM function
        use_llm_for_extraction=True,
        use_llm_for_questions=True
    )


# --------------------------
# Evaluation
# --------------------------

def normalize_label(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def accuracy_topk(gold: List[str], pred: List[Dict[str, Any]], k: int = 1) -> float:
    correct = 0
    for g, p in zip(gold, pred):
        g_norm = normalize_label(g)
        topk: List[str] = []
        probs = p.get("probs")
        if isinstance(probs, dict) and probs:
            topk = [lbl for lbl, _ in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:k]]
        else:
            if k >= 1 and "predicted_label" in p:
                topk = [p["predicted_label"]]
        if any(normalize_label(lbl) == g_norm for lbl in topk):
            correct += 1
    return correct / max(1, len(gold))

def simple_log_loss(gold: List[str], pred: List[Dict[str, Any]], eps: float = 1e-15) -> Optional[float]:
    losses = []
    for g, p in zip(gold, pred):
        probs = p.get("probs")
        if not isinstance(probs, dict) or not probs:
            return None
        g_norm = normalize_label(g)
        p_gold = 0.0
        for lbl, pr in probs.items():
            if normalize_label(lbl) == g_norm:
                p_gold = float(pr)
                break
        p_gold = min(1.0 - eps, max(eps, p_gold))
        losses.append(-math.log(p_gold))
    return sum(losses) / max(1, len(losses))


# --------------------------
# JSONL helpers
# --------------------------

def load_predictions_jsonl(path: str) -> List[Dict[str, Any]]:
    preds: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            preds.append(json.loads(line))
    return preds

def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", required=True, help="Path to patients CSV (test set). Must have EVIDENCES and a ground-truth label column.")
    ap.add_argument("--evidences", required=True, help="Path to evidences.json")
    ap.add_argument("--conditions", required=True, help="Path to conditions.json (not strictly used, but handy)")
    ap.add_argument("--label-col", default="GROUND_TRUTH", help="Column containing the gold diagnosis label")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N rows")
    ap.add_argument("--language", default="en", choices=["en", "fr"], help="Language for human-readable questions/values")
    ap.add_argument("--predictions", default=None, help="Optional path to predictions JSONL to evaluate (bypass LLM)")
    ap.add_argument("--out-dir", default="outputs", help="Directory to write artifacts")
    args = ap.parse_args()

    # Load metadata
    evidences_meta = load_json(args.evidences)
    _conditions = load_json(args.conditions)  # available for later restriction/logic if desired
    allowed_codes = set(evidences_meta.keys())

    # Load patients
    df = load_patients_csv(args.patients, limit=args.limit)
    if args.label_col not in df.columns:
        raise ValueError(f"Ground truth column '{args.label_col}' not found. Available: {list(df.columns)}")

    # Build per-patient evidence dicts (code-level + human-readable)
    code_dicts, human_dicts = build_patient_evidence_dicts(
        df,
        evidences_meta=evidences_meta,
        allowed_codes=allowed_codes,
        language=args.language,
    )

    # Build nested payloads with demographics separated
    patient_payloads: List[Dict[str, Any]] = []
    for i in range(len(human_dicts)):
        demo = {}
        if "AGE" in df.columns:
            age_val = df.iloc[i]["AGE"]
            if pd.notna(age_val):
                try:
                    demo["AGE"] = int(age_val)
                except Exception:
                    demo["AGE"] = age_val
        if "SEX" in df.columns:
            sex_val = df.iloc[i]["SEX"]
            if pd.notna(sex_val):
                demo["SEX"] = str(sex_val)

        patient_payloads.append({
            "row_index": i,
            "demographics": demo,
            "evidence": human_dicts[i],
            # optionally include raw codes for downstream consumers:
            # "codes": code_dicts[i],
        })

    # --- Preview before LLM ---
    print_preview_dicts(patient_payloads, n=100, title="patient payloads (demographics + evidence)")

    # Save artifacts
    os.makedirs(args.out_dir, exist_ok=True)
    nested_jsonl = os.path.join(args.out_dir, "patient_payloads.jsonl")
    human_jsonl  = os.path.join(args.out_dir, "patient_evidence_human.jsonl")
    code_jsonl   = os.path.join(args.out_dir, "patient_evidence_codes.jsonl")
    save_jsonl(nested_jsonl, patient_payloads)
    save_jsonl(human_jsonl, human_dicts)
    save_jsonl(code_jsonl, code_dicts)
    print(f"Wrote JSONL artifacts to:\n  {nested_jsonl}\n  {human_jsonl}\n  {code_jsonl}")

    # Ground truth labels
    gold_labels: List[str] = df[args.label_col].astype(str).tolist()

    # Predictions (either file or LLM)
    if args.predictions:
        predictions = load_predictions_jsonl(args.predictions)
        if len(predictions) != len(df):
            raise ValueError(f"Predictions count ({len(predictions)}) doesn't match patient rows ({len(df)}).")
    else:
        predictions = run_llm_pipeline(patient_payloads)

    # Save predictions for record
    preds_jsonl = os.path.join(args.out_dir, "predictions.jsonl")
    save_jsonl(preds_jsonl, predictions)
    print(f"Wrote predictions JSONL to: {preds_jsonl}")

    # Evaluate
    top1 = accuracy_topk(gold_labels, predictions, k=1)
    top3 = accuracy_topk(gold_labels, predictions, k=3)
    ll = simple_log_loss(gold_labels, predictions)

    print("\n=== Evaluation (test set) ===")
    print(f"Top-1 accuracy: {top1:.4f}")
    print(f"Top-3 accuracy: {top3:.4f}")
    if ll is not None:
        print(f"Log loss: {ll:.4f}")
    else:
        print("Log loss: N/A (predictions missing 'probs')")


if __name__ == "__main__":
    main()
