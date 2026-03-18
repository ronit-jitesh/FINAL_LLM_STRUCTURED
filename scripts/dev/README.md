# Internal Development Scripts

These scripts are **internal tooling** used during development and result verification. They are **NOT** part of the main evaluation pipeline and should not be run as part of reproducing the project results.

| Script | Purpose | Status |
|--------|---------|--------|
| `patch_v5_unknowns.py` | Resolved 30 unknown labels in hybrid_v5_results.csv via majority vote of DeBERTa predictions. | Already applied — results committed |
| `recompute_all.py` | Recomputes all metrics from source CSVs and patches the report markdown. Use after updating any results CSV. | Run as needed |
| `verify_and_update_report.py` | Verifies numbers in the report match the CSVs. Internal audit tool. | Run as needed |

These were separated from `src/` to make clear they are post-hoc tooling, not part of the numbered pipeline (`src/01_` through `src/10_`).

The patch applied to hybrid_v5_results.csv is fully documented in Section 9.5 (Limitations) of `NLI_Comprehensive_Results.md`.
