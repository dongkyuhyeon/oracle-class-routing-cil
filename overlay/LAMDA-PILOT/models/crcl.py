"""
CRCL: Codebook Routing Continual Learning

Architecture:
  - Frozen ViT backbone
  - ETF codebook: K anchors maximally separated in embedding space
  - Per-sample hard routing: cos_sim(frozen_feature, codebook) -> argmax -> LoRA_k
  - O-LoRA style forgetting prevention: orthogonality loss between current lora_A and
    snapshots of lora_A saved after each past task (no gradient projection)
  - Two-stage training (SLCA-style):
      Stage 1: train active LoRA + fc on new task data
      Stage 2: Classifier Alignment via LoRA-modified class mean sampling
"""

import logging
import math
from collections import defaultdict
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Normal, Independent
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.inc_net import CRCLNet
from utils.toolkit import tensor2numpy
from utils.oracle_routing import oracle_class_topk
from backbone.vit_lora_crcl import LoRAQKV, set_lora_indices

num_workers = 8


def _kmeans(X: torch.Tensor, k: int, n_iters: int = 50, spherical: bool = False) -> torch.Tensor:
    """
    Lloyd's K-means with K-means++ init. X: [N, D]. Returns [k, D] centroids.
    spherical=True: normalise centroids each step (consistent with cosine-similarity routing).
    """
    N = X.shape[0]
    gen = torch.Generator(device=X.device)
    gen.manual_seed(0)

    # K-means++ initialisation: spread initial centroids
    first = torch.randint(N, (1,), generator=gen, device=X.device).item()
    centroids = [X[first]]
    for _ in range(k - 1):
        c = torch.stack(centroids)  # [chosen, D]
        if spherical:
            # min cosine distance = 1 - max dot product
            sims = (X @ c.T).max(dim=1).values          # [N]
            dists = (1.0 - sims).clamp(min=0.0)
        else:
            dists = torch.cdist(X, c).min(dim=1).values  # [N]
        probs = dists / (dists.sum() + 1e-8)
        idx = torch.multinomial(probs, 1, generator=gen).item()
        centroids.append(X[idx])
    centroids = torch.stack(centroids)  # [k, D]
    if spherical:
        centroids = F.normalize(centroids, dim=1)

    for _ in range(n_iters):
        assigns = (X @ centroids.T).argmax(dim=1) if spherical \
            else torch.cdist(X, centroids).argmin(dim=1)
        new_c = torch.zeros_like(centroids)
        for j in range(k):
            m = assigns == j
            new_c[j] = X[m].mean(0) if m.any() else centroids[j]
        if spherical:
            new_c = F.normalize(new_c, dim=1)
        if (new_c - centroids).norm() < 1e-6:
            break
        centroids = new_c
    return centroids


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = CRCLNet(args)
        self.args = args

        self.batch_size   = args["batch_size"]
        self.init_lr      = args["init_lr"]
        self.weight_decay = args.get("weight_decay", 5e-4)
        self.min_lr       = args.get("min_lr", 1e-8)
        self.init_epochs  = args["init_epochs"]
        self.ca_epochs    = args.get("ca_epochs", 30)
        self.ca_lr        = args.get("ca_lr", 0.005)
        self.num_sampled_per_cls = args.get("num_sampled_per_cls", 256)
        self.top_k_loras = args.get("top_k_loras", 1)
        self.routing_mode = args.get("routing_mode", "learned")
        if self.routing_mode not in {"learned", "oracle_class"}:
            raise ValueError(
                "routing_mode must be 'learned' or 'oracle_class', got {}".format(
                    self.routing_mode))
        if not 1 <= self.top_k_loras <= self._network.num_loras:
            raise ValueError(
                "top_k_loras must be in [1, num_loras], got {} for K={}".format(
                    self.top_k_loras, self._network.num_loras))

        # Pluggable CL backend: "lite" (default, current lightweight mechanism),
        # "o-lora" / "inflora" / "sd-lora" (full paper fidelity, see vit_lora_crcl.py)
        self.backend_name = args.get("backend", "lite")

        # Ablation: growing-backend structure (top-k routing over the full
        # codebook) but with generation growth disabled after task 0 -- one
        # LoRA generation trains continuously across every task, never frozen,
        # never expanded. (aux_loss/orth naturally stays 0: every backend's
        # aux_loss is a no-op below 2 generations.)
        self.single_generation = args.get("single_generation", False)

        # O-LoRA style (lite backend only): snapshots of lora_A after each task,
        # used for orthogonality loss. Training-time-only, never used at inference.
        self._old_lora_snapshots: list = []
        self.orth_lambda = args.get(
            "orth_lambda", 0.5 if self.backend_name == "o-lora" else 0.1)

        if self.backend_name == "inflora":
            from backbone.vit_lora_crcl import InfLoRAProjector
            self._inflora_projector = InfLoRAProjector(
                lamb=args.get("inflora_lamb", 0.95),
                lame=args.get("inflora_lame", 1.0))

        # Class statistics (LoRA-modified features) for CA stage
        # dict: class_idx -> {lora_k: {'mean': np.array, 'cov': torch.Tensor}}
        self._cls_stats: dict = {}

        # Which LoRA indices each class routes to (for per-LoRA CA)
        # dict: class_idx -> set of LoRA indices
        self._cls_lora_routing: dict = {}

    def _route_topk(self, net, inputs, targets):
        """Return [B, top_k_loras] routing indices for the configured policy.

        learned:
            Frozen ViT features are matched to the ETF/codebook anchors.
        oracle_class:
            Ground-truth class labels define a deterministic, class-consistent
            and balanced cyclic assignment. For K=5, top-1 maps class c to
            c % 5; top-2 maps it to [c % 5, (c + 1) % 5].
        """
        if self.routing_mode == "oracle_class":
            return oracle_class_topk(
                targets.to(self._device), net.num_loras, self.top_k_loras)

        frozen_feats = net.extract_vector(inputs)
        return net.route_topk(frozen_feats, self.top_k_loras)

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # Build data loaders
        train_set = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        test_set = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        proto_set = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="test")

        self.train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        self.proto_loader = DataLoader(
            proto_set, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._network.to(self._device)

        # Pre-compute per-sample top-K routing for all training data + log distribution
        routing_map = self._precompute_routing()

        # Faithful backends: freeze last task's generation, grow a fresh trainable one
        # (must happen before stage-1 training since InfLoRA needs it to derive this
        # task's frozen A from this task's own routed data).
        if self.backend_name != "lite":
            self._prepare_generation(data_manager, routing_map)

        # Stage 1: per-sample routing training (routing from cache)
        self._stage1_train(routing_map)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # Compute LoRA-modified class statistics (per-sample routing)
        self._compute_class_stats(data_manager)

        # Stage 2: Classifier Alignment
        if self._cur_task > 0 and self.ca_epochs > 0:
            self._stage2_ca()

        # Lite backend only: save lora_A snapshot for orthogonality loss in future tasks
        if self.backend_name == "lite":
            self._save_lora_snapshot(
                self._network if not isinstance(self._network, nn.DataParallel)
                else self._network.module
            )

    # ------------------------------------------------------------------
    # Codebook initialisation (task 0)
    # ------------------------------------------------------------------

    def _init_codebook_kmeans(self):
        """Replace ETF codebook with K-means centroids from task 0's frozen features."""
        net = self._network if not isinstance(self._network, nn.DataParallel) \
            else self._network.module
        net.eval()

        all_feats = []
        with torch.no_grad():
            for _, inputs, _ in self.train_loader:
                feats = net.extract_vector(inputs.to(self._device))
                all_feats.append(feats.cpu())
        all_feats = torch.cat(all_feats, dim=0)  # [N, D]

        feats_norm = F.normalize(all_feats, dim=1)
        centroids = _kmeans(feats_norm, k=net.num_loras, spherical=True)  # [K, D]
        net.reset_codebook(centroids)
        logging.info("Task 0: codebook initialised via K-means (K={})".format(net.num_loras))

    # ------------------------------------------------------------------
    # LoRA splitting
    # ------------------------------------------------------------------

    def _check_and_split(self):
        """
        Before each task (t>0): check cumulative routing imbalance
        (past tasks + new task combined) and split overloaded LoRAs.
        Threshold: >1.5/K of all cumulative samples (= 1.5× uniform average).
        Past counts are redistributed proportionally on each split.
        """
        net = self._network if not isinstance(self._network, nn.DataParallel) \
            else self._network.module
        net.eval()

        # Extract new task's frozen features
        all_feats = []
        with torch.no_grad():
            for _, inputs, _ in self.train_loader:
                feats = net.extract_vector(inputs.to(self._device))
                all_feats.append(feats.cpu())
        all_feats = torch.cat(all_feats, dim=0).to(self._device)  # [N_new, D]

        indices = net.route(all_feats)  # [N_new]
        N_new = len(indices)
        K = net.num_loras
        past_total = sum(self._lora_sample_counts.values())
        grand_total = past_total + N_new

        def _cum_fracs(indices, K):
            new_cnt = {k: (indices == k).sum().item() for k in range(K)}
            return {
                k: (self._lora_sample_counts.get(k, 0) + new_cnt.get(k, 0)) / grand_total
                for k in range(K)
            }

        threshold = 1.5 / K  # 1.5× uniform average

        fracs = _cum_fracs(indices, K)
        frac_str = "  ".join(
            "LoRA{}: {:.1f}%".format(k, fracs[k] * 100) for k in range(K))
        logging.info("Task {}: cumulative routing fracs (K={}): {}".format(
            self._cur_task, K, frac_str))

        split_targets = [k for k in range(K) if fracs[k] > threshold]

        if not split_targets:
            logging.info("Task {}: no splits needed (threshold={:.0f}%)".format(
                self._cur_task, threshold * 100))
            return

        logging.info("Task {}: splitting LoRAs {} (>{:.0f}% threshold)".format(
            self._cur_task, split_targets, threshold * 100))

        # Process in reverse index order so lower indices stay stable after appends
        for k in sorted(split_targets, reverse=True):
            feats_k = all_feats[indices == k]
            if feats_k.shape[0] < 2:
                continue

            feats_norm = F.normalize(feats_k, dim=1).cpu()

            # k_split = ceil(current_frac / threshold): minimum k to bring each child below threshold
            k_split = max(2, math.ceil(fracs[k] / threshold))
            sub_anchors = _kmeans(feats_norm, k=k_split, spherical=True)
            assigns = (feats_norm @ sub_anchors.T).argmax(dim=1)
            n_sub = [(assigns == j).sum().item() for j in range(k_split)]
            logging.info("Task {}: LoRA{} split k={} (frac={:.1f}%)".format(
                self._cur_task, k, k_split, fracs[k] * 100))

            # Redistribute past cumulative counts proportionally
            old_cum = self._lora_sample_counts.get(k, 0)
            new_slot_start = net.num_loras
            k_split = len(n_sub)
            total_sub = sum(n_sub)

            net.split_anchor(k, sub_anchors.to(self._device))

            self._lora_sample_counts[k] = int(old_cum * n_sub[0] / total_sub)
            for j in range(1, k_split):
                self._lora_sample_counts[new_slot_start + j - 1] = int(old_cum * n_sub[j] / total_sub)

            # Re-route for remaining split targets
            indices = net.route(all_feats)

        logging.info("Task {}: K {} -> {}".format(self._cur_task, K, net.num_loras))

    # ------------------------------------------------------------------
    # Pre-compute routing for all training samples
    # ------------------------------------------------------------------

    def _precompute_routing(self) -> dict:
        """Extract frozen features once, compute per-sample top-K routing, log distribution."""
        from collections import Counter
        net = self._network if not isinstance(self._network, nn.DataParallel) \
            else self._network.module
        net.eval()

        routing_map = {}  # sample_idx -> list of lora_idx (length top_k_loras)
        with torch.no_grad():
            for sample_indices, inputs, targets in self.train_loader:
                inputs = inputs.to(self._device)
                topk_idx = self._route_topk(net, inputs, targets)  # [B, K]
                for s_idx, k_list in zip(sample_indices.tolist(), topk_idx.tolist()):
                    routing_map[s_idx] = k_list

        # Log top-1 distribution for readability
        top1_list = [v[0] for v in routing_map.values()]
        dist = Counter(top1_list)
        total = len(top1_list)
        dist_str = "  ".join(
            "LoRA{}: {}({:.0f}%)".format(k, v, 100 * v / total)
            for k, v in sorted(dist.items()))
        logging.info(
            "Task {} routing={} dist (top-1, current task): {}".format(
                self._cur_task, self.routing_mode, dist_str))

        return routing_map

    # ------------------------------------------------------------------
    # Stage 1: train active LoRA + fc
    # ------------------------------------------------------------------

    def _stage1_train(self, routing_map: dict):
        net = self._network if not isinstance(self._network, nn.DataParallel) \
            else self._network.module

        optimizer = optim.AdamW(
            [
                {"params": net.get_lora_params(), "lr": self.init_lr},
                {"params": net.get_fc_params(),   "lr": self.init_lr},
            ],
            betas=(0.0, 0.999),
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.init_epochs, eta_min=self.min_lr)

        prog = tqdm(range(self.init_epochs))
        for epoch in prog:
            self._network.train()
            total_loss, correct, total = 0.0, 0, 0

            for sample_indices, inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                local_targets = targets - self._known_classes

                optimizer.zero_grad()

                # Lookup pre-computed top-K routing
                # routing_map[i] = [k1, k2, ...] (length top_k_loras)
                sample_ids = sample_indices.tolist()
                B = inputs.size(0)

                # Build expanded batch: each sample appears top_k_loras times
                K = self.top_k_loras
                lora_assign, sample_map, is_top1 = [], [], []
                for i, sid in enumerate(sample_ids):
                    for rank, k in enumerate(routing_map[sid]):
                        lora_assign.append(k)
                        sample_map.append(i)
                        is_top1.append(rank == 0)

                lora_idx_t   = torch.tensor(lora_assign, dtype=torch.long, device=self._device)
                sample_map_t = torch.tensor(sample_map,  dtype=torch.long, device=self._device)
                is_top1_t    = torch.tensor(is_top1,     dtype=torch.bool,  device=self._device)

                expanded_inputs  = inputs[sample_map_t]         # [B*K, C, H, W]
                expanded_targets = local_targets[sample_map_t]  # [B*K]

                # Single forward with per-sample LoRA dispatch
                set_lora_indices(net.backbone, lora_idx_t)
                features = net.backbone(expanded_inputs)         # [B*K, D]
                set_lora_indices(net.backbone, None)

                # Compute losses per LoRA, sum for single backward
                loras_in_batch = list(set(lora_assign))
                per_lora_losses = {}
                for k in loras_in_batch:
                    mask_k   = (lora_idx_t == k)
                    logits_k = net.fc_list[k](features[mask_k])[:, self._known_classes:]
                    loss_k   = F.cross_entropy(logits_k, expanded_targets[mask_k])
                    per_lora_losses[k] = (loss_k, mask_k.sum().item())

                loss_sum  = sum(v[0] for v in per_lora_losses.values())
                batch_loss = sum(v[0].item() * v[1] for v in per_lora_losses.values())

                aux_loss = self._aux_loss(net, loras_in_batch)
                (loss_sum + self.orth_lambda * aux_loss).backward()

                # Accuracy: top-1 LoRA per sample
                with torch.no_grad():
                    top1_feats   = features[is_top1_t]                     # [B, D]
                    top1_k_list  = [routing_map[sid][0] for sid in sample_ids]
                    batch_correct = 0
                    for k in set(top1_k_list):
                        idx = [i for i, k1 in enumerate(top1_k_list) if k1 == k]
                        idx_t = torch.tensor(idx, device=self._device)
                        logits = net.fc_list[k](top1_feats[idx_t])[:, self._known_classes:]
                        batch_correct += logits.max(1)[1].eq(local_targets[idx_t]).sum().item()

                batch_seen = B

                optimizer.step()
                total_loss += batch_loss / B
                correct += batch_correct
                total += batch_seen

            scheduler.step()
            train_acc = 100.0 * correct / total
            prog.set_description(
                "Task {} Epoch {}/{} Loss {:.3f} Acc {:.2f}".format(
                    self._cur_task, epoch + 1, self.init_epochs,
                    total_loss / len(self.train_loader), train_acc))

        logging.info(prog.format_dict.get("postfix", ""))

    # ------------------------------------------------------------------
    # Compute class statistics (LoRA-modified features)
    # ------------------------------------------------------------------

    def _compute_class_stats(self, data_manager):
        """Recompute per-LoRA stats for ALL seen classes using current LoRA weights.
        Each class gets stats per LoRA it routes to (top-K routing).
        Structure: _cls_stats[class_idx] = {lora_k: {'mean': np.array, 'var': Tensor}}
        (diagonal covariance — per-dim variance only, not the full D x D matrix)
        """
        net = self._network if not isinstance(self._network, nn.DataParallel) \
            else self._network.module
        net.eval()

        for class_idx in range(0, self._total_classes):
            _, _, dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1), source="train", mode="test", ret_data=True)
            loader = DataLoader(dset, batch_size=self.batch_size, shuffle=False, num_workers=4)

            per_lora_feats = defaultdict(list)
            lora_routing: set = set()

            with torch.no_grad():
                for _, inputs, targets in loader:
                    inputs = inputs.to(self._device)
                    topk = self._route_topk(net, inputs, targets)  # [B, K]
                    lora_routing.update(topk.flatten().tolist())

                    for ki in range(self.top_k_loras):
                        k_indices = topk[:, ki]
                        for k in k_indices.unique():
                            k = k.item()
                            mask = (k_indices == k)
                            feats_k = net(inputs[mask], active_lora=k)["features"]
                            per_lora_feats[k].append(feats_k.cpu())

            cls_stats = {}
            for k, feats_list in per_lora_feats.items():
                feats = torch.nan_to_num(torch.cat(feats_list, dim=0), nan=0.0)
                mean = feats.mean(0).numpy()
                N, D = feats.shape
                if N >= 2:
                    var = feats.var(dim=0, unbiased=True)
                    # Clip per-dim variance: small-N classes give a noisy
                    # estimate whose largest dims can be pathologically
                    # large; unclipped, CA sampling from it can diverge
                    # fc_list[k] to NaN. Same rationale as the old
                    # eigenvalue clip, just per-dimension since the
                    # covariance is now diagonal.
                    cap = var.median().clamp(min=1e-6) * 10
                    var = var.clamp(min=1e-6, max=cap)
                else:
                    var = torch.full((D,), 1e-4)
                cls_stats[k] = {"mean": mean, "var": var}

            self._cls_stats[class_idx] = cls_stats
            self._cls_lora_routing[class_idx] = lora_routing

        lora_class_counts = defaultdict(int)
        for lora_set in self._cls_lora_routing.values():
            for k in lora_set:
                lora_class_counts[k] += 1
        cls_count_str = "  ".join(
            "LoRA{}: {}cls".format(k, lora_class_counts[k])
            for k in sorted(lora_class_counts))
        logging.info("Task {} per-LoRA class coverage (top-{} routing, {} classes seen): {}".format(
            self._cur_task, self.top_k_loras, self._total_classes, cls_count_str))

    # ------------------------------------------------------------------
    # Stage 2: Classifier Alignment (SLCA-style)
    # ------------------------------------------------------------------

    def _stage2_ca(self):
        """Per-LoRA Classifier Alignment.
        Each fc_k is calibrated independently using only the classes that
        route to LoRA_k — no cross-LoRA feature mixing, no staleness from
        other LoRAs' feature spaces.
        """
        net = self._network if not isinstance(self._network, nn.DataParallel) \
            else self._network.module
        net.eval()

        n = self.num_sampled_per_cls

        for k in range(net.num_loras):
            # Classes with at least one sample routing to LoRA_k
            classes_k = [c for c in range(self._total_classes)
                         if k in self._cls_lora_routing.get(c, set())]
            if not classes_k:
                continue

            optimizer_k = optim.SGD(
                net.fc_list[k].parameters(), lr=self.ca_lr, momentum=0.9,
                weight_decay=self.weight_decay)
            scheduler_k = optim.lr_scheduler.CosineAnnealingLR(
                optimizer_k, T_max=self.ca_epochs, eta_min=self.min_lr)

            for epoch in range(self.ca_epochs):
                sampled_feats, sampled_labels = [], []
                for c_id in classes_k:
                    stat = self._cls_stats[c_id][k]  # per-LoRA stats
                    dist = Independent(Normal(
                        torch.tensor(stat["mean"], dtype=torch.float32).to(self._device),
                        stat["var"].sqrt().to(self._device)), 1)
                    sampled_feats.append(dist.sample((n,)))
                    sampled_labels.extend([c_id] * n)

                sampled_feats  = torch.cat(sampled_feats, dim=0).to(self._device)
                sampled_labels = torch.tensor(sampled_labels, dtype=torch.long).to(self._device)

                perm = torch.randperm(sampled_feats.size(0))
                sampled_feats  = sampled_feats[perm]
                sampled_labels = sampled_labels[perm]

                total_loss = 0.0
                bs = min(256, sampled_feats.size(0))
                for start in range(0, sampled_feats.size(0), bs):
                    inp = sampled_feats[start: start + bs]
                    tgt = sampled_labels[start: start + bs]
                    logits = net(inp, active_lora=k, fc_only=True)["logits"]
                    loss = F.cross_entropy(logits, tgt)
                    if not torch.isfinite(loss):
                        # Diverged batch (rare, low-N classes) — skip the
                        # update rather than poisoning fc_list[k] with NaN.
                        continue
                    optimizer_k.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        net.fc_list[k].parameters(), max_norm=5.0)
                    optimizer_k.step()
                    total_loss += loss.item()

                scheduler_k.step()

        test_acc = self._compute_accuracy(self._network, self.test_loader)
        logging.info("CA Task {} (per-LoRA) Test {:.2f}".format(
            self._cur_task, test_acc))

    # ------------------------------------------------------------------
    # Faithful backend generation growth (o-lora / inflora / sd-lora)
    # ------------------------------------------------------------------

    def _prepare_generation(self, data_manager, routing_map: dict):
        """Freeze the generation trained last task (if any) and append a fresh
        trainable generation for every GrowingLoRAQKV block. InfLoRA additionally
        needs this task's per-slot input-feature covariance, computed first."""
        from backbone.vit_lora_crcl import GrowingLoRAQKV

        net = self._network if not isinstance(self._network, nn.DataParallel) \
            else self._network.module

        active_slots = sorted(set(k for klist in routing_map.values() for k in klist))
        blocks = [b for b in net.backbone.blocks if isinstance(b.attn.qkv, GrowingLoRAQKV)]

        if self.single_generation and self._cur_task > 0:
            n_gens = len(blocks[0].attn.qkv.gens_q) if blocks else 0
            logging.info(
                "Task {}: single_generation=True, skipping generation growth "
                "(generations per slot stays at {})".format(self._cur_task, n_gens))
            return

        if self.backend_name == "inflora":
            self._compute_inflora_covariance(net, blocks, routing_map, active_slots)

        for i, block in enumerate(blocks):
            qkv = block.attn.qkv
            if self._cur_task > 0:
                qkv.backend.on_task_end(qkv.gens_q, qkv.gens_v)

            if self.backend_name == "inflora":
                context = (self._inflora_projector, i)
            elif self.backend_name == "cl-lora":
                context = self._cur_task
            else:
                context = None
            new_gen_q = qkv.backend.new_generation(
                net.num_loras, qkv.rank, qkv.d_in, active_slots, context=context).to(self._device)
            new_gen_v = qkv.backend.new_generation(
                net.num_loras, qkv.rank, qkv.d_in, active_slots, context=context).to(self._device)
            qkv.gens_q.append(new_gen_q)
            qkv.gens_v.append(new_gen_v)

            if self.backend_name == "inflora":
                for k in active_slots:
                    self._inflora_projector.update_subspace(
                        i, k, self._cur_task, data_manager.nb_tasks)

        n_gens = len(blocks[0].attn.qkv.gens_q) if blocks else 0
        logging.info("Task {}: backend={} generations per slot = {}".format(
            self._cur_task, self.backend_name, n_gens))

    def _compute_inflora_covariance(self, net, blocks, routing_map: dict, active_slots: list):
        """No-grad forward pass over this task's data; accumulates per-(block, slot)
        input-feature covariance in self._inflora_projector, masked by top-k routing."""
        from backbone.vit_lora_crcl import set_active_lora

        self._inflora_projector.reset_accum()

        layer_acts = {}
        hooks = []

        def make_hook(idx):
            def hook(module, inp, out):
                layer_acts[idx] = inp[0].detach()
            return hook

        for i, block in enumerate(blocks):
            hooks.append(block.attn.qkv.register_forward_hook(make_hook(i)))

        net.eval()
        set_active_lora(net.backbone, -1)
        with torch.no_grad():
            for sample_indices, inputs, _ in self.train_loader:
                inputs = inputs.to(self._device)
                net.backbone(inputs)
                sids = sample_indices.tolist()
                for i in range(len(blocks)):
                    x = layer_acts[i]  # [B, N, D]
                    for bi, sid in enumerate(sids):
                        xs = x[bi]  # [N, D]
                        for k in routing_map.get(sid, []):
                            if k in active_slots:
                                self._inflora_projector.accumulate(i, k, xs)

        for h in hooks:
            h.remove()

    def _aux_loss(self, net, loras_in_batch: list) -> torch.Tensor:
        """Dispatches to the lite backend's snapshot-based orth loss, or the
        active faithful backend's own aux_loss (0 for inflora/sd-lora)."""
        if self.backend_name == "lite":
            return self._compute_orth_loss(net, loras_in_batch)

        from backbone.vit_lora_crcl import GrowingLoRAQKV
        loss = torch.tensor(0.0, device=self._device)
        for block in net.backbone.blocks:
            qkv = block.attn.qkv
            if isinstance(qkv, GrowingLoRAQKV):
                loss = loss + qkv.backend.aux_loss(qkv.gens_q, qkv.gens_v, loras_in_batch)
        return loss

    # ------------------------------------------------------------------
    # O-LoRA snapshot & orthogonality loss (lite backend)
    # ------------------------------------------------------------------

    def _save_lora_snapshot(self, net):
        """Save lora_A_q and lora_A_v from all blocks after each task."""
        snapshot = []
        for block in net.backbone.blocks:
            if isinstance(block.attn.qkv, LoRAQKV):
                snapshot.append({
                    "A_q": block.attn.qkv.lora_A_q.data.clone().cpu(),
                    "A_v": block.attn.qkv.lora_A_v.data.clone().cpu(),
                })
        self._old_lora_snapshots.append(snapshot)

    def _compute_orth_loss(self, net, loras_in_batch: list) -> torch.Tensor:
        """O-LoRA style orthogonality loss.
        Penalises overlap between current lora_A and each past task's lora_A snapshot.
        Formula: sum over past snapshots, blocks, active LoRAs of |old_A[k] @ cur_A[k].T|
        """
        if not self._old_lora_snapshots:
            return torch.tensor(0.0, device=self._device)

        blocks = [b for b in net.backbone.blocks if isinstance(b.attn.qkv, LoRAQKV)]
        orth_loss = torch.tensor(0.0, device=self._device)

        for snapshot in self._old_lora_snapshots:
            for block, snap in zip(blocks, snapshot):
                old_Aq = snap["A_q"].to(self._device)  # [K, rank, D]
                old_Av = snap["A_v"].to(self._device)
                cur_Aq = block.attn.qkv.lora_A_q       # [K, rank, D]
                cur_Av = block.attn.qkv.lora_A_v
                for k in loras_in_batch:
                    # [rank, D] @ [D, rank] → [rank, rank], sum of abs values
                    orth_loss = orth_loss + torch.abs(old_Aq[k] @ cur_Aq[k].T).sum()
                    orth_loss = orth_loss + torch.abs(old_Av[k] @ cur_Av[k].T).sum()

        # Normalise so lambda is independent of block count, LoRA count, task count
        n_terms = len(self._old_lora_snapshots) * len(blocks) * len(loras_in_batch) * 2
        return orth_loss / n_terms

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _eval_cnn(self, loader):
        """Per-sample hard routing at inference. Logs per-LoRA accuracy."""
        net = self._network if not isinstance(self._network, nn.DataParallel) \
            else self._network.module
        net.eval()

        y_pred, y_true = [], []
        lora_correct = defaultdict(int)
        lora_total   = defaultdict(int)

        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device)

                # Step 1: learned feature routing or GT-class Oracle routing
                topk_indices = self._route_topk(net, inputs, targets)  # [B, K]

                # Step 2: sum logits from all K LoRAs each sample routes to
                logits_out = torch.zeros(
                    inputs.size(0), self._total_classes, device=self._device)

                for k in range(net.num_loras):
                    mask = (topk_indices == k).any(dim=1)
                    if not mask.any():
                        continue
                    logits_out[mask] += net(inputs[mask], active_lora=k)["logits"]

                # Per-LoRA accuracy tracking (top-1 routing for comparison)
                top1_indices = topk_indices[:, 0]
                for k in range(net.num_loras):
                    mask = (top1_indices == k)
                    if not mask.any():
                        continue
                    preds_k = logits_out[mask].argmax(dim=1).cpu()
                    lora_correct[k] += preds_k.eq(targets[mask.cpu()]).sum().item()
                    lora_total[k]   += mask.sum().item()

                preds = torch.topk(logits_out, k=self.topk, dim=1)[1]
                y_pred.append(preds.cpu().numpy())
                y_true.append(targets.numpy())

        # Log per-LoRA accuracy
        lora_acc_str = "  ".join(
            "LoRA{}: {:.1f}%({})".format(
                k, 100.0 * lora_correct[k] / lora_total[k], lora_total[k])
            for k in sorted(lora_total))
        logging.info("Task {} per-LoRA test acc: {}".format(self._cur_task, lora_acc_str))

        return np.concatenate(y_pred), np.concatenate(y_true)

    def _compute_accuracy(self, model, loader):
        net = model if not isinstance(model, nn.DataParallel) else model.module
        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device)
                topk_indices = self._route_topk(net, inputs, targets)  # [B, K]

                logits_out = torch.zeros(
                    inputs.size(0), self._total_classes, device=self._device)
                for k in range(net.num_loras):
                    mask = (topk_indices == k).any(dim=1)
                    if not mask.any():
                        continue
                    logits_out[mask] += net(inputs[mask], active_lora=k)["logits"]

                preds = logits_out.argmax(dim=1)
                correct += (preds.cpu() == targets).sum().item()
                total += targets.size(0)
        return np.around(100.0 * correct / total, decimals=2)

