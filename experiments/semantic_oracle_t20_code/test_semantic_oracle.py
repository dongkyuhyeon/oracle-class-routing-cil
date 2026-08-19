from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).with_name("semantic_oracle.py")
SPEC = importlib.util.spec_from_file_location("semantic_oracle", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
SemanticOracleRouter = MODULE.SemanticOracleRouter
load_semantic_mapping = MODULE.load_semantic_mapping


def make_mapping(path: Path):
    classes = []
    for label_id in range(200):
        classes.append(
            {
                "label_id": label_id,
                "wnid": f"n{label_id:08d}",
                "class_name": f"class-{label_id}",
                "expert_id": label_id // 40,
                "train_samples": 1,
                "task_id": label_id // 10,
            }
        )
    path.write_text(json.dumps({"metadata": {}, "classes": classes}), encoding="utf-8")


class SemanticOracleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mapping_path = Path(self.tmp.name) / "mapping.json"
        make_mapping(self.mapping_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_mapping_has_exact_balance(self):
        mapping, _ = load_semantic_mapping(str(self.mapping_path), 5, 200)
        self.assertEqual(mapping.numel(), 200)
        self.assertEqual(torch.bincount(mapping, minlength=5).tolist(), [40] * 5)

    def test_same_label_always_routes_same_expert(self):
        router = SemanticOracleRouter(str(self.mapping_path), 5, 200)
        targets = torch.tensor([7, 7, 7, 80, 80, 199])
        routes_a = router.route(targets)
        routes_b = router.route(targets)
        self.assertTrue(torch.equal(routes_a, routes_b))
        self.assertEqual(routes_a.shape, (6, 1))

    def test_expected_experts(self):
        router = SemanticOracleRouter(str(self.mapping_path), 5, 200)
        targets = torch.tensor([0, 39, 40, 79, 80, 119, 120, 159, 160, 199])
        expected = torch.tensor([[0], [0], [1], [1], [2], [2], [3], [3], [4], [4]])
        self.assertTrue(torch.equal(router.route(targets), expected))

    def test_missing_path_fails_clearly(self):
        with self.assertRaises(FileNotFoundError):
            SemanticOracleRouter("/definitely/missing/mapping.json", 5, 200)

    def test_invalid_low_label_fails(self):
        router = SemanticOracleRouter(str(self.mapping_path), 5, 200)
        with self.assertRaises(ValueError):
            router.route(torch.tensor([-1]))

    def test_invalid_high_label_fails(self):
        router = SemanticOracleRouter(str(self.mapping_path), 5, 200)
        with self.assertRaises(ValueError):
            router.route(torch.tensor([200]))

    def test_reload_is_identical(self):
        a, _ = load_semantic_mapping(str(self.mapping_path), 5, 200)
        b, _ = load_semantic_mapping(str(self.mapping_path), 5, 200)
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
