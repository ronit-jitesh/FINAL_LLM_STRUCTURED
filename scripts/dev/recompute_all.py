#!/usr/bin/env python3
"""
Master recompute script for the NLI project.
Reruns analysis and figure generation to ensure all artifacts are synchronized.
"""
import subprocess
import os
import sys

def run_script(path):
    print(f"\n>>> Running {path}...")
    result = subprocess.run([sys.executable, path], capture_output=False)
    if result.returncode != 0:
        print(f"!!! Error running {path}")
        return False
    return True

def main():
    # Ensure we are in the project root
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.chdir(root)
    
    scripts = [
        "src/06_cost_analysis.py",
        "src/07a_figures_main.py",
        "src/07b_figure2_pareto.py",
        "src/08_error_analysis.py",
        "src/09_genre_label_analysis.py",
        "src/10_significance_tests.py"
    ]
    
    for s in scripts:
        if not run_script(s):
            sys.exit(1)
            
    print("\n✅ All artifacts synchronized.")

if __name__ == "__main__":
    main()
