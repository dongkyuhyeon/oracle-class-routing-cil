"""Generate deterministic balanced semantic expert mapping for ImageNet-A T=20, K=5.

Run from R-LoRA/LAMDA-PILOT after copying this file into scripts/ or invoking it
with --rlo-root. This script uses ImageFolder WNIDs, DataManager's actual
incremental label order (shuffle=True, seed=1993), WordNet Wu-Palmer similarity,
and capacity-constrained medoid clustering with exactly 40 classes per expert.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from torchvision import datasets
from nltk.corpus import wordnet as wn


NUM_CLASSES = 200
NUM_EXPERTS = 5
CAPACITY = 40
SEED = 1993
INIT_CLS = 10
INCREMENT = 10
MAX_ITERS = 100


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rlo-root", type=Path, default=Path.cwd())
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def wnid_to_synset(wnid: str):
    if len(wnid) < 2:
        raise ValueError(f"Invalid WNID: {wnid}")
    return wn.synset_from_pos_and_offset(wnid[0], int(wnid[1:]))


def semantic_distance(a, b) -> float:
    sim = a.wup_similarity(b)
    if sim is None:
        return 1.0
    return 1.0 - float(sim)


def pairwise_distances(synsets):
    n = len(synsets)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = semantic_distance(synsets[i], synsets[j])
            D[i, j] = d
            D[j, i] = d
    return D


def choose_initial_medoids(D, wnids, k=NUM_EXPERTS):
    # Deterministic first medoid: class with maximum total distance.
    totals = D.sum(axis=1)
    max_total = totals.max()
    candidates = np.where(np.isclose(totals, max_total))[0].tolist()
    first = min(candidates, key=lambda i: wnids[i])
    selected = [first]

    # Deterministic farthest-first; tie -> smaller WNID.
    while len(selected) < k:
        best_idx = None
        best_score = None
        for i in range(len(wnids)):
            if i in selected:
                continue
            score = min(D[i, m] for m in selected)
            if best_score is None or score > best_score + 1e-12:
                best_score = score
                best_idx = i
            elif abs(score - best_score) <= 1e-12 and wnids[i] < wnids[best_idx]:
                best_idx = i
        selected.append(best_idx)
    return selected


def balanced_assign(D, medoids):
    slots = np.repeat(np.arange(NUM_EXPERTS), CAPACITY)
    cost = np.empty((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for c in range(NUM_CLASSES):
        for slot_idx, expert_id in enumerate(slots):
            cost[c, slot_idx] = D[c, medoids[expert_id]]

    # Tiny deterministic tie-break that does not change semantic ordering.
    eps = 1e-12
    cost += eps * np.arange(NUM_CLASSES)[None, :]

    rows, cols = linear_sum_assignment(cost)
    assignment = np.full(NUM_CLASSES, -1, dtype=np.int64)
    for r, c in zip(rows, cols):
        assignment[r] = slots[c]
    return assignment


def update_medoids(D, assignment, wnids):
    medoids = []
    for expert_id in range(NUM_EXPERTS):
        members = np.where(assignment == expert_id)[0]
        if len(members) != CAPACITY:
            raise RuntimeError(
                f"Expert {expert_id} has {len(members)} classes; expected {CAPACITY}."
            )
        sums = D[np.ix_(members, members)].sum(axis=1)
        best = sums.min()
        candidates = members[np.where(np.isclose(sums, best))[0]].tolist()
        medoids.append(min(candidates, key=lambda i: wnids[i]))
    return medoids


def cluster_balanced(D, wnids):
    medoids = choose_initial_medoids(D, wnids)
    assignment = None
    converged = False
    iterations = 0

    for iteration in range(MAX_ITERS):
        iterations = iteration + 1
        new_assignment = balanced_assign(D, medoids)
        new_medoids = update_medoids(D, new_assignment, wnids)

        if assignment is not None and np.array_equal(new_assignment, assignment) and new_medoids == medoids:
            assignment = new_assignment
            medoids = new_medoids
            converged = True
            break

        assignment = new_assignment
        medoids = new_medoids

    return assignment, medoids, converged, iterations


def class_name_for_synset(synset):
    names = synset.lemma_names()
    return names[0].replace("_", " ") if names else synset.name()


def main():
    args = parse_args()
    rlo_root = args.rlo_root.resolve()
    pilot = rlo_root / "LAMDA-PILOT" if (rlo_root / "LAMDA-PILOT").is_dir() else rlo_root

    if not (pilot / "utils" / "data_manager.py").is_file():
        raise FileNotFoundError(f"R-LoRA LAMDA-PILOT not found at: {pilot}")

    sys.path.insert(0, str(pilot))
    from utils.data_manager import DataManager

    train_dir = pilot / "data" / "imagenet-a" / "train"
    test_dir = pilot / "data" / "imagenet-a" / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"ImageNet-A directories not found: train={train_dir}, test={test_dir}"
        )

    try:
        wn.ensure_loaded()
    except LookupError as e:
        raise RuntimeError(
            "NLTK WordNet corpus is not installed. Do not create a fallback grouping; "
            "install/download WordNet first."
        ) from e

    train_dset = datasets.ImageFolder(str(train_dir))
    test_dset = datasets.ImageFolder(str(test_dir))

    if train_dset.classes != test_dset.classes:
        raise RuntimeError("Train/test ImageFolder class order differs.")
    if len(train_dset.classes) != NUM_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_CLASSES} ImageNet-A classes, got {len(train_dset.classes)}."
        )

    dm_args = {"dataset": "imageneta", "model_name": "crcl"}
    dm = DataManager(
        dataset_name="imageneta",
        shuffle=True,
        seed=SEED,
        init_cls=INIT_CLS,
        increment=INCREMENT,
        args=dm_args,
    )

    order = list(dm._class_order)
    if len(order) != NUM_CLASSES or sorted(order) != list(range(NUM_CLASSES)):
        raise RuntimeError("Unexpected DataManager class order.")

    original_to_incremental = {
        original_idx: incremental_label
        for incremental_label, original_idx in enumerate(order)
    }

    train_counts_original = Counter(train_dset.targets)
    test_counts_original = Counter(test_dset.targets)

    # Cluster in original ImageFolder/WNID index space.
    wnids = list(train_dset.classes)
    synsets = [wnid_to_synset(w) for w in wnids]
    D = pairwise_distances(synsets)
    assignment, medoids, converged, iterations = cluster_balanced(D, wnids)

    counts = np.bincount(assignment, minlength=NUM_EXPERTS)
    if counts.tolist() != [CAPACITY] * NUM_EXPERTS:
        raise RuntimeError(f"Unbalanced assignment: {counts.tolist()}")

    rows = []
    for original_idx, wnid in enumerate(wnids):
        label_id = original_to_incremental[original_idx]
        rows.append(
            {
                "label_id": int(label_id),
                "wnid": wnid,
                "class_name": class_name_for_synset(synsets[original_idx]),
                "expert_id": int(assignment[original_idx]),
                "train_samples": int(train_counts_original[original_idx]),
                "test_samples": int(test_counts_original[original_idx]),
                "task_id": int(label_id // INCREMENT),
                "original_imagefolder_label": int(original_idx),
            }
        )

    rows.sort(key=lambda r: r["label_id"])

    expert_summary = []
    for expert_id in range(NUM_EXPERTS):
        members = [r for r in rows if r["expert_id"] == expert_id]
        expert_summary.append(
            {
                "expert_id": expert_id,
                "class_count": len(members),
                "train_samples": sum(r["train_samples"] for r in members),
                "test_samples": sum(r["test_samples"] for r in members),
                "medoid_wnid": wnids[medoids[expert_id]],
                "medoid_class_name": class_name_for_synset(synsets[medoids[expert_id]]),
            }
        )

    payload = {
        "metadata": {
            "dataset": "imageneta",
            "num_classes": NUM_CLASSES,
            "num_experts": NUM_EXPERTS,
            "classes_per_expert": CAPACITY,
            "seed": SEED,
            "shuffle": True,
            "init_cls": INIT_CLS,
            "increment": INCREMENT,
            "semantic_metric": "WordNet Wu-Palmer similarity",
            "distance": "1 - wup_similarity",
            "none_similarity_handling": "distance=1.0",
            "initialization": "deterministic farthest-first medoids",
            "balanced_assignment": "scipy.optimize.linear_sum_assignment with 40 slots per expert",
            "max_iterations": MAX_ITERS,
            "iterations": iterations,
            "converged": converged,
        },
        "expert_summary": expert_summary,
        "classes": rows,
    }

    output = args.output or (pilot / "exps" / "semantic_class_to_expert_ina_t20_k5.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Saved: {output}")
    for row in expert_summary:
        print(
            f"Expert {row['expert_id']}: {row['class_count']} classes | "
            f"train={row['train_samples']} | test={row['test_samples']} | "
            f"medoid={row['medoid_wnid']} ({row['medoid_class_name']})"
        )
    print(f"Total: {len(rows)} unique classes")
    print(f"Duplicate: {len(rows) - len({r['label_id'] for r in rows})}")
    print(f"Missing: {NUM_CLASSES - len({r['label_id'] for r in rows})}")


if __name__ == "__main__":
    main()
