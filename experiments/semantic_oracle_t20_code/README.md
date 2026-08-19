# ImageNet-A T20 Semantic Oracle Top-1

This folder contains only the code for the first Semantic Oracle experiment.

## Purpose

Partition the 200 ImageNet-A classes into five semantic experts with exactly 40 classes per expert using WordNet, then use the GT incremental class label to select exactly one LoRA expert during both training and evaluation.

This does **not** evaluate the learned feature/codebook router.

## Files

- `generate_semantic_mapping.py` — builds the deterministic balanced WordNet mapping.
- `semantic_oracle.py` — loads the mapping and routes GT labels to one expert.
- `oracle_ina_t20_semantic_top1.json` — fixed T=20 / K=5 / lite / seed=1993 config.
- `test_semantic_oracle.py` — mapping and routing unit tests.
- `crcl_semantic_oracle_patch.diff` — minimal CRCL integration patch; learned routing remains preserved.

## Fixed experiment

- Dataset: ImageNet-A (`imageneta`)
- 200 classes
- T=20 (`init_cls=10`, `increment=10`)
- K=5 experts
- Semantic Oracle Top-1
- backend=`lite`
- rank=8
- memory=0
- batch size=64
- 20 epochs/task
- seed=1993
- backbone=`pretrained_vit_b16_224_in21k`
- AdamW, lr=0.001, weight decay=0.0005
- CosineAnnealingLR
- CA off (`ca_epochs=0`)
- orthogonal loss off (`orth_lambda=0`)

## Mapping rule

1. Read the actual ImageNet-A train/test `ImageFolder` WNIDs.
2. Reproduce the R-LoRA `DataManager` incremental label order with `shuffle=true`, seed 1993.
3. Convert each WNID to a WordNet synset.
4. Compute pairwise Wu-Palmer distance: `1 - wup_similarity`.
5. If WordNet returns `None`, use maximum distance `1.0`.
6. Initialize five medoids deterministically with farthest-first selection.
7. Give each expert exactly 40 slots.
8. Use `scipy.optimize.linear_sum_assignment` for capacity-constrained assignment.
9. Recompute each medoid as the class minimizing total intra-cluster distance.
10. Iterate until stable or 100 iterations.

No CLIP, LLM embedding, text embedding, random K-means, or `class_id % 5` grouping is used.

## Server integration order

Copy the files into an existing R-LoRA checkout without replacing unrelated files:

```text
semantic_oracle.py
  -> LAMDA-PILOT/utils/semantic_oracle.py

generate_semantic_mapping.py
  -> LAMDA-PILOT/scripts/generate_semantic_mapping.py

oracle_ina_t20_semantic_top1.json
  -> LAMDA-PILOT/exps/oracle_ina_t20_semantic_top1.json
```

Apply `crcl_semantic_oracle_patch.diff` only after inspecting the current `crcl.py`, because server checkouts may already contain unrelated local modifications.

Then generate the mapping once, run tests/smoke test, and only after they pass run:

```bash
python main.py --config=./exps/oracle_ina_t20_semantic_top1.json
```

The generated mapping file is:

```text
LAMDA-PILOT/exps/semantic_class_to_expert_ina_t20_k5.json
```
