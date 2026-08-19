import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "overlay"
    / "LAMDA-PILOT"
    / "utils"
    / "oracle_routing.py"
)
SPEC = importlib.util.spec_from_file_location("oracle_routing", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
oracle_class_topk = MODULE.oracle_class_topk


class OracleRoutingTest(unittest.TestCase):
    def test_top1_is_balanced_for_imagenet_a(self):
        routes = oracle_class_topk(torch.arange(200), num_loras=5, top_k=1)
        counts = torch.bincount(routes.flatten(), minlength=5)
        self.assertEqual(routes.shape, (200, 1))
        self.assertEqual(counts.tolist(), [40, 40, 40, 40, 40])

    def test_top2_is_balanced_for_imagenet_a(self):
        routes = oracle_class_topk(torch.arange(200), num_loras=5, top_k=2)
        counts = torch.bincount(routes.flatten(), minlength=5)
        self.assertEqual(routes.shape, (200, 2))
        self.assertEqual(counts.tolist(), [80, 80, 80, 80, 80])

    def test_same_class_always_has_same_route(self):
        targets = torch.tensor([7, 7, 7, 12, 12])
        routes = oracle_class_topk(targets, num_loras=5, top_k=2)
        self.assertTrue(torch.equal(routes[0], routes[1]))
        self.assertTrue(torch.equal(routes[1], routes[2]))
        self.assertTrue(torch.equal(routes[3], routes[4]))

    def test_invalid_top_k_fails(self):
        with self.assertRaises(ValueError):
            oracle_class_topk(torch.arange(5), num_loras=5, top_k=6)


if __name__ == "__main__":
    unittest.main()

