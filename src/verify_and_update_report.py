#!/usr/bin/env python3
"""
verify_and_update_report.py
============================
1. Verifies hybrid_v5_results.csv is clean (800/400, 0 unknowns)
2. Computes exact F1 and accuracy
3. Prints every line in NLI_Comprehensive_Results.md that mentions v5
   so you know exactly what to update

Run:
    cd "/Users/ronitjitesh/Downloads/LLM Final"
    python src/verify_and_update_report.py
"""

import os
import re
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
REPORT_PATH = os.path.join(PROJECT_DIR, "NLI_Comprehensive_Results.md")
README_PATH = os.path.join(PROJECT_DIR, "README.md")

LABELS = ["entailment", "neutral", "contradiction"]

# ── 1. Verify CSV ─────────────────────────────────────────
print("=" * 60)
print("STEP 1 — VERIFY hybrid_v5_results.csv")
print("=" * 60)

df = pd.read_csv(os.path.join(RESULTS_DIR, "hybrid_v5_results.csv"))

results = {}
all_ok = True

for set_name in ["matched", "mismatched"]:
    expected_n = 800 if set_name == "matched" else 400
    sub = df[df["set"] == set_name]
    n = len(sub)
    unknowns = (sub["label_pred"] == "unknown").sum()
    acc = accuracy_score(sub["label_true"], sub["label_pred"])
    f1  = f1_score(sub["label_true"], sub["label_pred"],
                   average="macro", labels=LABELS)
    
    status = "✅" if (n == expected_n and unknowns == 0) else "❌"
    print(f"\n  {set_name} {status}")
    print(f"    N        : {n}/{expected_n}")
    print(f"    Accuracy : {acc*100:.2f}%")
    print(f"    Macro F1 : {f1:.4f}")
    print(f"    Unknowns : {unknowns}")
    
    if n != expected_n or unknowns != 0:
        all_ok = False
    
    results[set_name] = {"n": n, "acc": acc, "f1": f1, "unknowns": unknowns}

# Source breakdown
print("\n  Source breakdown:")
print(df["source"].value_counts().to_string())

if not all_ok:
    print("\n❌ ISSUES FOUND — do not update report until fixed")
    exit(1)

print("\n✅ CSV is clean — safe to update report")

# ── 2. Compute exact replacement strings ─────────────────
m_acc  = results["matched"]["acc"]
m_f1   = results["matched"]["f1"]
mm_acc = results["mismatched"]["acc"]
mm_f1  = results["mismatched"]["f1"]

print("\n" + "=" * 60)
print("STEP 2 — EXACT NEW NUMBERS")
print("=" * 60)
print(f"\n  Hybrid v5 Matched   Acc = {m_acc*100:.1f}%   F1 = {m_f1:.4f}")
print(f"  Hybrid v5 Mismatched Acc = {mm_acc*100:.1f}%   F1 = {mm_f1:.4f}")

# ── 3. Find all v5 lines in report ───────────────────────
print("\n" + "=" * 60)
print("STEP 3 — LINES IN REPORT THAT MENTION v5 / ENSEMBLE")
print("=" * 60)

with open(REPORT_PATH, "r") as f:
    report_lines = f.readlines()

v5_lines = []
for i, line in enumerate(report_lines, 1):
    if re.search(r"v5|[Ee]nsemble [Gg]ate|[Ee]nsemble [Gg]ating|89\.5|90\.25|89\.50|0\.895|0\.902", line):
        v5_lines.append((i, line.rstrip()))

print(f"\n  Found {len(v5_lines)} relevant lines:\n")
for lineno, text in v5_lines:
    print(f"  Line {lineno:4d}: {text}")

# ── 4. Print exact report replacements ───────────────────
print("\n" + "=" * 60)
print("STEP 4 — WHAT TO CHANGE IN NLI_Comprehensive_Results.md")
print("=" * 60)

m_acc_str  = f"{m_acc*100:.1f}%"
mm_acc_str = f"{mm_acc*100:.1f}%"

print(f"""
  FIND:    89.50%  or  89.5%     → REPLACE WITH: {m_acc_str}
  FIND:    90.25%  or  90.3%     → REPLACE WITH: {mm_acc_str}
  FIND:    $0.288                → REPLACE WITH: (recompute from cost_summary.csv)
  
  In tables, the Hybrid v5 row should read:
    | Hybrid v5 (Ensemble) | {m_acc_str} | {mm_acc_str} | 12.5% | $0.288 |
  
  In Executive Summary, update:
    "89.5% accuracy"  →  "{m_acc_str} accuracy"
  
  In §5.6 or §9.1, any mention of v5 accuracy needs updating.
  
  In §9.5 Limitations, ADD this sentence:
    "30 unknown labels in hybrid v5 (caused by verbose CoT parse 
     failures) were resolved post-hoc: 19 via API retry and 11 via 
     majority vote of the three DeBERTa encoder predictions."
""")

# ── 5. Check README too ───────────────────────────────────
print("=" * 60)
print("STEP 5 — LINES IN README.md THAT MENTION v5")
print("=" * 60)

with open(README_PATH, "r") as f:
    readme_lines = f.readlines()

for i, line in enumerate(readme_lines, 1):
    if re.search(r"v5|[Ee]nsemble|89\.5|90\.25", line):
        print(f"  Line {i:3d}: {line.rstrip()}")

print("\n" + "=" * 60)
print("DONE — update the lines shown above, then run:")
print("  python src/06_cost_analysis.py")
print("  python src/07a_figures_main.py")
print("  python src/07b_figure2_pareto.py")
print("  python src/09_genre_label_analysis.py")
print("=" * 60)
