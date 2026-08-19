# Experiment records

Every run created by `scripts/run_oracle_class_routing.sh` is stored under:

```text
experiments/runs/YYYYMMDD_HHMMSS_<experiment>/
├── 1_config.json
├── 2_command.sh
├── 3_git_commit_sha.txt
├── 4_metrics.json
├── 5_expert_assignments.csv
├── 6_class_coverage_and_accuracy.csv
├── 7_full.log
├── 8_graphs/
└── 9_summary.md
```

Only these nine experiment artifacts are kept. Datasets, checkpoints, and
pretrained weights are intentionally excluded.

The collector automatically extracts:

- Final Top-1, Average Top-1, and Final Top-5
- cumulative top-1 training assignments per expert
- final test assignments and per-expert accuracy
- final class coverage per expert
- accuracy-curve and expert-overview graphs when matplotlib is available

`9_summary.md` contains an automatic numerical summary plus dedicated sections
for the interpretation and the next experiment decision.
