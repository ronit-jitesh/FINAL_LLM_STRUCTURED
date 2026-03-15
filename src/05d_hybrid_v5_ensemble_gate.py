#!/usr/bin/env python3
"""
Notebook 05d — Hybrid v5: 3-DeBERTa Ensemble Gate + GPT-4o P4 (CoT)
======================================================================
Gate logic: Run DeBERTa-v3-small, DeBERTa-v3-base, DeBERTa-v3-large in parallel.
  - If ALL THREE agree → use their unanimous prediction (free, ~95% accurate)
  - If ANY disagree   → escalate to GPT-4o P4 (CoT few-shot)

Why this works:
  - ~87.5% of samples are unanimous → high accuracy, zero API cost
  - Only ~12.5% of samples escalate → LLM handles genuinely hard cases

Prerequisite:
  02_encoder_baselines.py must have been run with all 5 models.
  encoder_predictions_matched.csv must contain all deberta columns.

Outputs:
  results/hybrid_v5_results.csv

PATCH (2026-03): Added resume logic. If hybrid_v5_results.csv already exists,
  only rows not yet processed are re-run. This allows recovery from interrupted runs.
"""

import os
import re
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

load_dotenv()

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEB_MODELS = ["deberta_v3_small", "deberta_v3_base", "deberta_v3_large"]
OUT_PATH   = os.path.join(RESULTS_DIR, "hybrid_v5_results.csv")

# ============================================================
# Label parser — CoT-aware
# ============================================================
def parse_label(text):
    if not text:
        return "unknown"
    text_lower = text.lower()
    label_match = re.search(r'label\s*:\s*(contradiction|entailment|neutral)', text_lower)
    if label_match:
        return label_match.group(1)
    first_line = text_lower.strip().split("\n")[0].strip()
    first_line_clean = re.sub(r"[^a-z]", " ", first_line).strip()
    for label in ["contradiction", "entailment", "neutral"]:
        if first_line_clean.startswith(label):
            return label
    last_pos   = -1
    last_label = "unknown"
    for label in ["contradiction", "entailment", "neutral"]:
        pos = text_lower.rfind(label)
        if pos > last_pos:
            last_pos   = pos
            last_label = label
    return last_label


# ============================================================
# GPT-4o P4 (CoT few-shot)
# ============================================================
PROMPT_P4 = (
    "Classify the natural language inference relationship step by step.\n\n"
    "Examples:\n"
    'Premise: "The concert was held outdoors."\n'
    'Hypothesis: "The event took place inside a building."\n'
    "Step-by-step: The premise says outdoor; the hypothesis says inside. These directly contradict.\n"
    "Label: contradiction\n\n"
    'Premise: "She completed her PhD in linguistics."\n'
    'Hypothesis: "She has a doctoral degree."\n'
    "Step-by-step: A PhD is a doctoral degree. The hypothesis follows necessarily.\n"
    "Label: entailment\n\n"
    'Premise: "The report was published in March."\n'
    'Hypothesis: "The author spent years writing it."\n'
    "Step-by-step: Publication date says nothing about how long writing took.\n"
    "Label: neutral\n\n"
    "Now classify:\n"
    "Premise: {premise}\n"
    "Hypothesis: {hypothesis}\n"
    "Step-by-step:"
)

def call_gpt4o_p4(premise, hypothesis, max_retries=3):
    from openai import OpenAI
    client = OpenAI()
    prompt = PROMPT_P4.format(premise=premise, hypothesis=hypothesis)
    INPUT_COST  = 2.50
    OUTPUT_COST = 10.00
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=120,
                seed=42,
            )
            raw   = response.choices[0].message.content.strip()
            label = parse_label(raw)
            usage = response.usage
            cost  = (usage.prompt_tokens  * INPUT_COST  / 1_000_000
                   + usage.completion_tokens * OUTPUT_COST / 1_000_000)
            return label, usage.prompt_tokens + usage.completion_tokens, cost
        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"  GPT-4o error (attempt {attempt+1}): {e}  — retrying in {wait}s")
            time.sleep(wait)
    return "unknown", 0, 0.0


# ============================================================
# Core hybrid v5 runner — with resume support
# ============================================================
def run_hybrid_v5(df_test, df_encoder, call_api_fn, set_name,
                  already_done_idx: set):
    """
    Hybrid v5 with resume: skips rows already in already_done_idx.
    Returns list of result dicts.
    """
    results    = []
    api_calls  = 0
    total_cost = 0.0

    pred_cols = [f"{m}_pred" for m in DEB_MODELS]
    conf_cols = [f"{m}_conf" for m in DEB_MODELS]

    for i in tqdm(range(len(df_test)), desc=f"Hybrid v5 [{set_name}]"):
        # Resume: skip rows already saved
        if (set_name, i) in already_done_idx:
            continue

        row     = df_test.iloc[i]
        enc_row = df_encoder.iloc[i]

        preds = [enc_row[col] for col in pred_cols]
        confs = [float(enc_row[col]) for col in conf_cols]

        unique_preds = set(preds)

        if len(unique_preds) == 1:
            final_pred = preds[0]
            avg_conf   = float(np.mean(confs))
            source     = "ensemble"
            tokens     = 0
            cost       = 0.0
        else:
            pred, tokens, cost = call_api_fn(row["premise"], row["hypothesis"])
            final_pred  = pred
            avg_conf    = float(np.mean(confs))
            source      = "api"
            api_calls  += 1
            total_cost += cost
            time.sleep(0.05)

        results.append({
            "idx"       : i,
            "hybrid"    : "v5_ensemble_gate",
            "set"       : set_name,
            "premise"   : row["premise"],
            "hypothesis": row["hypothesis"],
            "genre"     : row["genre"],
            "label_true": row["label_text"],
            "label_pred": final_pred,
            "source"    : source,
            "avg_conf"  : avg_conf,
            "deb_preds" : "|".join(preds),
            "tokens"    : tokens,
            "cost_usd"  : cost,
        })

        # Checkpoint every 50 rows
        if len(results) % 50 == 0:
            _checkpoint(results, set_name)

    print(f"  [{set_name}] New rows this run: {len(results)} | "
          f"API calls: {api_calls} | Cost: ${total_cost:.4f}")
    return results


def _checkpoint(new_rows, set_name):
    """Append new rows to the CSV safely."""
    df_new = pd.DataFrame(new_rows)
    if os.path.exists(OUT_PATH):
        df_existing = pd.read_csv(OUT_PATH)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        # Deduplicate on (set, idx) keeping last
        df_combined = df_combined.drop_duplicates(subset=["set", "idx"], keep="last")
        df_combined.to_csv(OUT_PATH, index=False)
    else:
        df_new.to_csv(OUT_PATH, index=False)


def print_metrics(df, set_name):
    sub = df[df["set"] == set_name]
    if sub.empty:
        return
    acc = accuracy_score(sub["label_true"], sub["label_pred"])
    f1  = f1_score(sub["label_true"], sub["label_pred"], average="macro",
                   labels=["entailment", "neutral", "contradiction"])
    ens_pct = (sub["source"] == "ensemble").mean() * 100
    api_pct = (sub["source"] == "api").mean() * 100
    cost    = sub["cost_usd"].sum()
    errors  = (sub["label_true"] != sub["label_pred"]).sum()
    print(f"\n  ── Hybrid v5 | {set_name} ({len(sub)} rows) ────────────────")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Macro F1 : {f1:.4f}")
    print(f"  Ensemble : {ens_pct:.1f}%  |  API: {api_pct:.1f}%")
    print(f"  Errors   : {errors}  |  Cost: ${cost:.4f}")


def main():
    # ── Load test data ───────────────────────────────────────
    df_test_m  = pd.read_csv(os.path.join(DATA_DIR, "nli_test_800.csv"))
    df_test_mm = pd.read_csv(os.path.join(DATA_DIR, "nli_test_mm_400.csv"))

    enc_m_path  = os.path.join(RESULTS_DIR, "encoder_predictions_matched.csv")
    enc_mm_path = os.path.join(RESULTS_DIR, "encoder_predictions_mm.csv")

    if not os.path.exists(enc_m_path):
        print("❌ Run 02_encoder_baselines.py first!")
        return

    df_enc_m  = pd.read_csv(enc_m_path)
    df_enc_mm = pd.read_csv(enc_mm_path) if os.path.exists(enc_mm_path) else None

    # ── Check required columns ───────────────────────────────
    required = ([f"{m}_pred" for m in DEB_MODELS] +
                [f"{m}_conf" for m in DEB_MODELS])
    missing = [c for c in required if c not in df_enc_m.columns]
    if missing:
        print(f"❌ Missing encoder columns: {missing}")
        return

    # ── Load existing results for resume ─────────────────────
    already_done = set()
    if os.path.exists(OUT_PATH):
        df_existing = pd.read_csv(OUT_PATH)
        already_done = set(zip(df_existing["set"], df_existing["idx"]))
        m_done  = len(df_existing[df_existing["set"] == "matched"])
        mm_done = len(df_existing[df_existing["set"] == "mismatched"])
        print(f"\n📂 Resuming from existing file:")
        print(f"   matched done   : {m_done}/800")
        print(f"   mismatched done: {mm_done}/400")
    else:
        print("\n📂 No existing results — starting fresh")

    print("\n" + "#" * 65)
    print("# HYBRID v5: 3-DeBERTa Ensemble Gate + GPT-4o P4 (CoT)")
    print("#" * 65)

    # ── Run matched ──────────────────────────────────────────
    m_new = run_hybrid_v5(df_test_m, df_enc_m, call_gpt4o_p4,
                          "matched", already_done)
    if m_new:
        _checkpoint(m_new, "matched")

    # ── Run mismatched ───────────────────────────────────────
    if df_enc_mm is not None:
        missing_mm = [c for c in required if c not in df_enc_mm.columns]
        if not missing_mm:
            mm_new = run_hybrid_v5(df_test_mm, df_enc_mm, call_gpt4o_p4,
                                   "mismatched", already_done)
            if mm_new:
                _checkpoint(mm_new, "mismatched")
        else:
            print(f"\n⚠️  Skipping mismatched — missing columns: {missing_mm}")

    # ── Final save & metrics ──────────────────────────────────
    df_final = pd.read_csv(OUT_PATH)
    df_final = df_final.drop_duplicates(subset=["set", "idx"], keep="last")
    df_final.to_csv(OUT_PATH, index=False)

    print(f"\n✅ Final CSV: {OUT_PATH}")
    print(f"   matched rows   : {len(df_final[df_final['set']=='matched'])}/800")
    print(f"   mismatched rows: {len(df_final[df_final['set']=='mismatched'])}/400")

    print_metrics(df_final, "matched")
    print_metrics(df_final, "mismatched")

    print("\n" + "=" * 65)
    print("HYBRID v5 COMPLETE ✅")
    print("=" * 65)


if __name__ == "__main__":
    main()
