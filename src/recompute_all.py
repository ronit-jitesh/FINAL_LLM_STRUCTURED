#!/usr/bin/env python3
"""
MASTER RECOMPUTE SCRIPT (Foolproof Version)
===========================================
Automates the synchronization of NLI_Comprehensive_Results.md and README.md
whenever hybrid results CSVs are updated.
"""

import os
import re
import pandas as pd
import numpy as np
import subprocess
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
REPORT_PATH = os.path.join(PROJECT_DIR, "NLI_Comprehensive_Results.md")
README_PATH = os.path.join(PROJECT_DIR, "README.md")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")

LABELS = ["entailment", "neutral", "contradiction"]

def get_metrics(filename, threshold=None, is_v5=False):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    if is_v5:
        sub = df[df["set"] == "matched"]
        mm_sub = df[df["set"] == "mismatched"]
    else:
        if "threshold" in df.columns:
            sub = df[(df["threshold"] == threshold) & (df["set"] == "matched")]
            mm_sub = df[(df["threshold"] == threshold) & (df["set"] == "mismatched")]
        else:
            sub = df[df["set"] == "matched"]
            mm_sub = df[df["set"] == "mismatched"]

    if len(sub) == 0: return None
    acc = accuracy_score(sub["label_true"], sub["label_pred"])
    p, r, f, s = precision_recall_fscore_support(sub["label_true"], sub["label_pred"], labels=LABELS, zero_division=0)
    
    metrics = {
        "acc": acc,
        "mm_acc": accuracy_score(mm_sub["label_true"], mm_sub["label_pred"]) if len(mm_sub) > 0 else 0.0,
        "macro_f1": float(np.mean(f)),
        "api_pct": float((sub["source"] == "api").mean() if "source" in sub.columns else 0.0),
        "cost_1k": float(sub["cost_usd"].mean() * 1000 if "cost_usd" in sub.columns else 0.0),
        "errors": int(len(sub) * (1 - acc)),
        "p": p, "r": r, "f1": f
    }
    if is_v5:
        api_rows = sub[sub["source"] == "api"]
        metrics["api_acc"] = accuracy_score(api_rows["label_true"], api_rows["label_pred"]) if len(api_rows) > 0 else 0.63
    return metrics

def patch_report(metrics_map):
    if not os.path.exists(REPORT_PATH): return
    with open(REPORT_PATH, "r") as f: lines = f.readlines()

    v1, v4, v5 = metrics_map["v1"], metrics_map["v4"], metrics_map["v5"]
    new_lines = []
    
    for i, line in enumerate(lines):
        clean_line = line.replace("**", "").replace("__", "")

        # --- EXECUTIVE SUMMARY ---
        if i < 25:
             if v5:
                 line = line.replace("91.0% matched and 92.5% mismatched accuracy", f"{v5['acc']*100:.1f}% matched and {v5['mm_acc']*100:.1f}% mismatched accuracy")
                 line = line.replace("only 63%", f"only {v5.get('api_acc', 0.63)*100:.0f}%")
             if v4:
                 line = re.sub(r'reached (\**)\d+\.\d+%\s*matched accuracy at (\**)\$\d+\.\d+', 
                               rf'reached \g<1>{v4["acc"]*100:.2f}% matched accuracy at \g<2>${v4["cost_1k"]:.3f}', line)

        # --- TABLES ---
        
        # 1. Patch v1
        if "Hybrid v1 theta=0.90" in clean_line and v1:
             line = f"| Hybrid v1 theta=0.90 | {v1['acc']*100:.1f}% | **{v1['mm_acc']*100:.1f}%** | {v1['api_pct']*100:.1f}% | ${v1['cost_1k']:.3f} | {v1['errors']} |\n"
        elif "theta=0.90" in clean_line and v1 and "### 5.2 Hybrid v1" in "".join(lines[max(0, i-10):i]):
             line = f"| **theta=0.90** | **{v1['acc']*100:.1f}%** | **{v1['mm_acc']*100:.1f}%** | {v1['api_pct']*100:.1f}% | ${v1['cost_1k']:.3f} | {v1['errors']} |\n"
        
        # 2. Patch v4
        elif "Hybrid v4 theta=0.90 [best cost]" in clean_line and v4:
             if "Macro F1" in clean_line:
                 line = f"| **Hybrid v4 theta=0.90 [best cost]** | {v4['acc']*100:.1f}% | {v4['macro_f1']:.3f} | {v4['p'][0]:.3f} / {v4['r'][0]:.3f} / {v4['f1'][0]:.3f} | {v4['p'][1]:.3f} / {v4['r'][1]:.3f} / {v4['f1'][1]:.3f} | {v4['p'][2]:.3f} / {v4['r'][2]:.3f} / {v4['f1'][2]:.3f} |\n"
             else:
                 line = f"| Hybrid v4 theta=0.90 [best cost] | **{v4['acc']*100:.1f}%** | {v4['mm_acc']*100:.1f}% | {v4['api_pct']*100:.1f}% | **${v4['cost_1k']:.3f}** | **{v4['errors']}** |\n"
        elif "theta=0.90" in clean_line and v4 and "### 5.5 Hybrid v4" in "".join(lines[max(0, i-10):i]):
             line = f"| **theta=0.90** | **{v4['acc']*100:.2f}%** | **{v4['mm_acc']*100:.2f}%** | **{v4['api_pct']*100:.1f}%** | **${v4['cost_1k']:.3f}** | **{v4['errors']}** |\n"

        # 3. Patch v4 generic row
        elif "Hybrid v4 (0.9)" in clean_line and v4:
             line = f"| Hybrid v4 (0.9)  | {v4['p'][0]:.3f} / {v4['r'][0]:.3f} / {v4['f1'][0]:.3f} | {v4['p'][1]:.3f} / {v4['r'][1]:.3f} / {v4['f1'][1]:.3f} | {v4['p'][2]:.3f} / {v4['r'][2]:.3f} / {v4['f1'][2]:.3f} |\n"

        # 4. Patch v5
        elif "| Results |" in line and v5 and "Hybrid v5" in "".join(lines[max(0, i-10):i]):
             line = f"| Results | **{v5['acc']*100:.1f}%** | **{v5['mm_acc']*100:.1f}%** | {100-v5['api_pct']*100:.1f}% | {v5['api_pct']*100:.1f}% | ${v5['cost_1k']:.3f} |\n"
        elif "Hybrid v5 (Ensemble) [best overall]" in clean_line and v5:
             line = f"| **Hybrid v5 (Ensemble) [best overall]** | **{v5['acc']*100:.1f}%** | **{v5['mm_acc']*100:.1f}%** | {v5['api_pct']*100:.1f}% | ${v5['cost_1k']:.3f} | {v5['errors']} |\n"
        elif "Hybrid v5 Ensemble" in clean_line and v5:
             if "Macro F1" in clean_line or (i > 300 and "|" in line):
                 line = f"| Hybrid v5 Ensemble | **{v5['acc']*100:.1f}%** | **{v5['macro_f1']:.3f}** | {v5['p'][0]:.3f} / {v5['r'][0]:.3f} / {v5['f1'][0]:.3f} | {v5['p'][1]:.3f} / {v5['r'][1]:.3f} / {v5['f1'][1]:.3f} | {v5['p'][2]:.3f} / {v5['r'][2]:.3f} / {v5['f1'][2]:.3f} |\n"
        elif "Hybrid v5 Ens" in clean_line and v5:
             line = f"| Hybrid v5 Ens    | {v5['p'][0]:.3f} / {v5['r'][0]:.3f} / {v5['f1'][0]:.3f} | {v5['p'][1]:.3f} / {v5['r'][1]:.3f} / {v5['f1'][1]:.3f} | {v5['p'][2]:.3f} / {v5['r'][2]:.3f} / {v5['f1'][2]:.3f} |\n"

        # 5. Patch escalated findings
        elif "scores only" in line and v5:
             line = re.sub(r'scores only (\**)\d+\.\d+%', rf'scores only \g<1>{v5.get("api_acc", 0.63)*100:.0f}%', line)
             line = re.sub(r'scores only \d+%', rf'scores only {v5.get("api_acc", 0.63)*100:.0f}%', line)
        elif "accuracy on escalated rows" in line and v5:
             line = re.sub(r'(\**)\d+%\s*accuracy on escalated rows', rf'\g<1>{v5.get("api_acc", 0.63)*100:.0f}% accuracy on escalated rows', line)

        new_lines.append(line)

    with open(REPORT_PATH, "w") as f: f.writelines(new_lines)
    print("✅ Patched NLI_Comprehensive_Results.md")

def patch_readme(metrics_map):
    if not os.path.exists(README_PATH): return
    with open(README_PATH, "r") as f: lines = f.readlines()
    v1, v4, v5 = metrics_map["v1"], metrics_map["v4"], metrics_map["v5"]
    new_lines = []
    for line in lines:
        clean_line = line.replace("**", "").replace("__", "")
        if "Hybrid v5 (Ensemble) ⭐ best" in clean_line and v5:
            line = f"| **Hybrid v5 (Ensemble) ⭐ best** | **{v5['acc']*100:.1f}%** | **{v5['mm_acc']*100:.1f}%** | {v5['api_pct']*100:.1f}% | ${v5['cost_1k']:.3f} |\n"
        elif "Hybrid v4 θ=0.90 ⭐ cost" in clean_line and v4:
            line = f"| Hybrid v4 θ=0.90 ⭐ cost | **{v4['acc']*100:.1f}%** | {v4['mm_acc']*100:.1f}% | {v4['api_pct']*100:.1f}% | **${v4['cost_1k']:.3f}** |\n"
        elif "Hybrid v1 θ=0.90" in clean_line and v1:
            line = f"| Hybrid v1 θ=0.90 | {v1['acc']*100:.1f}% | **{v1['mm_acc']*100:.1f}%** | {v1['api_pct']*100:.1f}% | ${v1['cost_1k']:.3f} |\n"
        new_lines.append(line)
    with open(README_PATH, "w") as f: f.writelines(new_lines)
    print("✅ Patched README.md")

def main():
    print("="*60 + "\nMASTER RECOMPUTE & SYNCHRONIZATION\n" + "="*60)
    print("\n[1/3] Computing fresh metrics...")
    m = {
        "v1": get_metrics("hybrid_v1_results.csv", threshold=0.9),
        "v4": get_metrics("hybrid_v4_results.csv", threshold=0.9),
        "v5": get_metrics("hybrid_v5_results.csv", is_v5=True)
    }
    for k, v in m.items():
        if v: print(f"  {k.upper()}: Acc={v['acc']*100:.2f}%, F1={v['macro_f1']:.3f}, Cost/1k=${v['cost_1k']:.3f}")
    print("\n[2/3] Patching report files...")
    patch_report(m); patch_readme(m)
    print("\n[3/3] Regenerating figures and derived data...")
    if os.path.exists(FIGURES_DIR):
        for f in os.listdir(FIGURES_DIR):
            if f.endswith(".png"):
                try: os.remove(os.path.join(FIGURES_DIR, f))
                except: pass
    scripts = ["src/06_cost_analysis.py", "src/07a_figures_main.py", "src/07b_figure2_pareto.py", "src/09_genre_label_analysis.py"]
    for s in scripts:
        print(f"  Running {s}...")
        subprocess.run(["python3", s], check=True, capture_output=True)
    print("\n" + "="*60 + "\nSYNCHRONIZATION COMPLETE ✅\n" + "="*60)

if __name__ == "__main__": main()
