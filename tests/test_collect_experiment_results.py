import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "collect_experiment_results.py"
SPEC = importlib.util.spec_from_file_location("collector", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SAMPLE_LOG = """
Task 0 routing=oracle_class dist (top-1, current task): LoRA0: 10(20%)  LoRA1: 10(20%)
Task 1 routing=oracle_class dist (top-1, current task): LoRA0: 12(24%)  LoRA1: 8(16%)
Task 19 per-LoRA class coverage (top-2 routing, 200 classes seen): LoRA0: 80cls  LoRA1: 80cls
Task 19 per-LoRA test acc: LoRA0: 55.5%(240)  LoRA1: 44.5%(160)
CNN top1 curve: [70.0, 60.0, 50.0]
CNN top5 curve: [90.0, 85.0, 80.0]
Average Accuracy (CNN): 60.0
"""


class CollectorParserTest(unittest.TestCase):
    def test_metrics_and_expert_stats(self):
        parsed = MODULE.parse_log(SAMPLE_LOG)
        self.assertEqual(parsed["final_top1"], 50.0)
        self.assertEqual(parsed["average_top1"], 60.0)
        self.assertEqual(parsed["final_top5"], 80.0)
        self.assertEqual(parsed["train_counts"], {0: 22, 1: 18})
        self.assertEqual(parsed["test_counts"], {0: 240, 1: 160})
        self.assertEqual(parsed["expert_accuracy"], {0: 55.5, 1: 44.5})
        self.assertEqual(parsed["class_coverage"], {0: 80, 1: 80})
        self.assertEqual(parsed["total_classes"], 200)

    def test_writes_requested_record_files(self):
        parsed = MODULE.parse_log(SAMPLE_LOG)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            MODULE.write_metrics(run_dir, parsed, 0)
            MODULE.write_expert_csv(run_dir, parsed)
            MODULE.write_coverage_csv(run_dir, parsed)
            MODULE.write_summary(run_dir, parsed, 0)
            self.assertTrue((run_dir / "4_metrics.json").is_file())
            self.assertTrue((run_dir / "5_expert_assignments.csv").is_file())
            self.assertTrue((run_dir / "6_class_coverage_and_accuracy.csv").is_file())
            self.assertTrue((run_dir / "9_summary.md").is_file())


if __name__ == "__main__":
    unittest.main()
