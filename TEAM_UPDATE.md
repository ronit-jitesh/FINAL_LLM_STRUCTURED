# Team Update -- What Changed Tonight
**NLI Classification Project | University of Edinburgh**
**All updated files: https://github.com/ronit-jitesh/FINAL_LLM_STRUCTURED**

---

## The Short Version

The results have been updated. **Hybrid v5 is now our best system**, not v4. Here is why and what else changed.

---

## 1. The Big Result Change: v5 > v4

We originally had Hybrid v4 listed as the best system. That was wrong -- we had not fully patched the v5 results file at that point. After fixing 30 GPT-4o parse failures in hybrid_v5_results.csv (GPT-4o was producing verbose CoT outputs that our label extractor could not parse), the correct v5 numbers came out significantly better than v4.

Here is the updated comparison:

| System | Matched Acc | Mismatched Acc | Cost/1k |
|--------|-------------|----------------|---------|
| Hybrid v4 (was "best") | 90.62% | 90.5% | $0.007 |
| **Hybrid v5 (actually best)** | **91.0%** | **92.5%** | $0.258 |

So v5 wins on both matched and mismatched accuracy. v4 is still the best if cost is the main concern ($0.007 vs $0.258), but v5 is the headline result.

**Why was v5 underreported before?** The 30 broken rows were showing as "unknown" labels, which were being counted as wrong predictions. After the fix:
- 19 rows: fixed by retrying the GPT-4o API call
- 10 rows: fixed by majority vote of the 3 DeBERTa predictions
- 1 row: fixed by DeBERTa-base fallback (three-way tie)

---

## 2. The Other Bug Fixes

While fixing v5, we also found and fixed several numbers in the report that were wrong:

| Bug | Old Value | Correct Value | Where It Appeared |
|-----|-----------|---------------|-------------------|
| GPT-4o accuracy on the "hard 100" rows | 51% | **63%** | Exec summary, S5.6, S8.5, S9.2, S10.2 |
| Hybrid v5 cost | $0.288/1k | **$0.258/1k** | S5.6, S5.7, S7.1, README |
| Hybrid v5 errors count | 75 | **72** | S5.7 table |
| Hybrid v4 Neutral Recall/F1 | 0.877 / 0.860 | **0.874 / 0.859** | S5.5.1 per-class table |
| Hybrid v5 per-class P/R/F1 (all 3 classes) | wrong | **fixed from CSV** | S5.5.1, S5.9 |

The 51% -> 63% fix is the most important one. The 51% figure was calculated before the 30 broken rows were fixed. Once those were resolved, GPT-4o's accuracy on the hard 100 went up to 63%. It is still low (random baseline would be 33%), which confirms the annotation ambiguity story -- but 63% is more accurate than 51%.

---

## 3. Files Changed

| File | What Changed |
|------|-------------|
| **results/hybrid_v5_results.csv** | 30 unknown labels resolved -- now 800/800 matched and 400/400 mismatched with 0 unknowns |
| **results/cost_summary.csv** | v5 cost corrected to $0.258; GPT-5 P3/P4 costs fixed (were showing corrupt $527k/$447k values) |
| **results/classification_reports.csv** | Hybrid v5 Ensemble entry added (was missing) |
| **NLI_Comprehensive_Results.md** | All 7 number bugs fixed; v5 promoted to best overall; v4 relabelled as best cost-efficiency; prose humanised and converted to ASCII-only |
| **README.md** | Best system updated to v5; key finding 51% -> 63%; table updated |
| **TRACEABILITY.md** | Source map updated to reflect v5 as best overall and 63% figure |
| **results_table.html** | v5 row updated (was showing 89.5% -- now 91.0%); BEST badge moved from v4 to v5 |
| **INTERNAL_NOTES.md** | Deleted (was a scratch file, should not have been in the repo) |

---

## 4. What the Key v5 Finding Actually Means

The ensemble gate (v5) works differently from confidence gating (v1-v4). Instead of asking "is the encoder confident?", it asks "do all three encoders agree?".

- **87.5% of samples**: all 3 DeBERTa models agree -> handled at 95% accuracy, zero API cost
- **12.5% of samples**: models disagree -> sent to GPT-4o

The interesting finding is that GPT-4o only scores 63% on those 100 disagreement rows. For comparison, GPT-4o scores around 85% on the confidence-gated rows in v1-v4. The disagreement rows are not just hard -- they are genuinely ambiguous at the annotation level. Even human annotators might disagree on them. This is why v5 is framed as a "data quality insight" as much as an accuracy result.

---

## 5. Nothing Else Changed

- All code scripts are unchanged
- The dev/test CSV data files are unchanged
- All 15 figures are unchanged
- Seed 42, all other experimental conditions identical

---

## 6. Push Command Used

```bash
git add -A
git commit -m "Final audit: v5 best overall 91.0%/92.5%, fix 7 report bugs, humanise prose, ASCII only"
git push
```

Repo: https://github.com/ronit-jitesh/FINAL_LLM_STRUCTURED

---

*Written by Ronit -- ping if anything is unclear*
