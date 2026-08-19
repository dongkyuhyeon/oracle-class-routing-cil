#!/usr/bin/env python3
"""Collect the nine persistent artifacts for one Oracle-routing experiment."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path


def _last_curve(text: str, name: str) -> list[float]:
    matches = re.findall(rf"{re.escape(name)}:\s*(\[[^\n]+\])", text)
    if not matches:
        return []
    try:
        values = ast.literal_eval(matches[-1])
        return [float(v) for v in values]
    except (SyntaxError, ValueError, TypeError):
        return []


def parse_log(text: str) -> dict:
    top1_curve = _last_curve(text, "CNN top1 curve")
    top5_curve = _last_curve(text, "CNN top5 curve")

    avg_matches = re.findall(
        r"Average Accuracy \(CNN\):\s*([0-9]+(?:\.[0-9]+)?)", text
    )
    average_top1 = float(avg_matches[-1]) if avg_matches else (
        sum(top1_curve) / len(top1_curve) if top1_curve else None
    )

    train_counts: dict[int, int] = {}
    train_lines = re.findall(
        r"Task\s+\d+\s+routing=.*?dist \(top-1, current task\):\s*([^\n]+)",
        text,
    )
    for line in train_lines:
        for expert, count in re.findall(r"LoRA(\d+):\s*(\d+)\(", line):
            idx = int(expert)
            train_counts[idx] = train_counts.get(idx, 0) + int(count)

    test_counts: dict[int, int] = {}
    expert_accuracy: dict[int, float] = {}
    test_lines = re.findall(r"per-LoRA test acc:\s*([^\n]+)", text)
    if test_lines:
        for expert, accuracy, count in re.findall(
            r"LoRA(\d+):\s*([0-9.]+)%\((\d+)\)", test_lines[-1]
        ):
            idx = int(expert)
            expert_accuracy[idx] = float(accuracy)
            test_counts[idx] = int(count)

    class_coverage: dict[int, int] = {}
    coverage_lines = re.findall(r"per-LoRA class coverage .*?([^\n]+)", text)
    if coverage_lines:
        for expert, count in re.findall(
            r"LoRA(\d+):\s*(\d+)cls", coverage_lines[-1]
        ):
            class_coverage[int(expert)] = int(count)

    total_classes = None
    total_class_matches = re.findall(
        r"per-LoRA class coverage .*?(\d+) classes seen", text
    )
    if total_class_matches:
        total_classes = int(total_class_matches[-1])

    experts = sorted(
        set(train_counts) | set(test_counts) | set(expert_accuracy) | set(class_coverage)
    )
    return {
        "top1_curve": top1_curve,
        "top5_curve": top5_curve,
        "final_top1": top1_curve[-1] if top1_curve else None,
        "average_top1": average_top1,
        "final_top5": top5_curve[-1] if top5_curve else None,
        "train_counts": train_counts,
        "test_counts": test_counts,
        "expert_accuracy": expert_accuracy,
        "class_coverage": class_coverage,
        "total_classes": total_classes,
        "experts": experts,
    }


def _ratio(count: int, total: int) -> float | None:
    return round(100.0 * count / total, 4) if total else None


def write_metrics(run_dir: Path, parsed: dict, exit_code: int) -> None:
    metrics = {
        "run_exit_code": exit_code,
        "final_top1_accuracy": parsed["final_top1"],
        "average_top1_accuracy": parsed["average_top1"],
        "final_top5_accuracy": parsed["final_top5"],
        "top1_curve": parsed["top1_curve"],
        "top5_curve": parsed["top5_curve"],
    }
    (run_dir / "4_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_expert_csv(run_dir: Path, parsed: dict) -> None:
    total_train = sum(parsed["train_counts"].values())
    total_test = sum(parsed["test_counts"].values())
    with (run_dir / "5_expert_assignments.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["expert", "train_samples", "train_ratio_pct", "test_samples", "test_ratio_pct"]
        )
        for expert in parsed["experts"]:
            train = parsed["train_counts"].get(expert, 0)
            test = parsed["test_counts"].get(expert, 0)
            writer.writerow(
                [expert, train, _ratio(train, total_train), test, _ratio(test, total_test)]
            )


def write_coverage_csv(run_dir: Path, parsed: dict) -> None:
    total_classes = parsed["total_classes"]
    with (run_dir / "6_class_coverage_and_accuracy.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["expert", "class_coverage", "coverage_ratio_pct", "expert_accuracy_pct"]
        )
        for expert in parsed["experts"]:
            coverage = parsed["class_coverage"].get(expert)
            coverage_ratio = (
                round(100.0 * coverage / total_classes, 4)
                if coverage is not None and total_classes
                else None
            )
            writer.writerow(
                [
                    expert,
                    coverage,
                    coverage_ratio,
                    parsed["expert_accuracy"].get(expert),
                ]
            )


def write_graphs(run_dir: Path, parsed: dict) -> None:
    graph_dir = run_dir / "8_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if parsed["top1_curve"] or parsed["top5_curve"]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if parsed["top1_curve"]:
            ax.plot(parsed["top1_curve"], marker="o", label="Top-1")
        if parsed["top5_curve"]:
            ax.plot(parsed["top5_curve"], marker="s", label="Top-5")
        ax.set_xlabel("Task")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("CIL accuracy curve")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(graph_dir / "accuracy_curve.png", dpi=160)
        plt.close(fig)

    if parsed["experts"]:
        experts = parsed["experts"]
        train = [parsed["train_counts"].get(k, 0) for k in experts]
        test = [parsed["test_counts"].get(k, 0) for k in experts]
        accuracy = [parsed["expert_accuracy"].get(k, 0.0) for k in experts]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        x = list(range(len(experts)))
        axes[0].bar([i - 0.2 for i in x], train, width=0.4, label="Train")
        axes[0].bar([i + 0.2 for i in x], test, width=0.4, label="Test")
        axes[0].set_xticks(x, [f"LoRA {k}" for k in experts])
        axes[0].set_title("Expert assignment")
        axes[0].legend()
        axes[1].bar([f"LoRA {k}" for k in experts], accuracy)
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Per-expert accuracy")
        fig.tight_layout()
        fig.savefig(graph_dir / "expert_overview.png", dpi=160)
        plt.close(fig)


def _fmt(value) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def write_summary(run_dir: Path, parsed: dict, exit_code: int) -> None:
    lines = [
        "# Experiment summary",
        "",
        "## Automatic result summary",
        "",
        f"- Run exit code: {exit_code}",
        f"- Final Top-1 Accuracy: {_fmt(parsed['final_top1'])}",
        f"- Average Top-1 Accuracy: {_fmt(parsed['average_top1'])}",
        f"- Final Top-5 Accuracy: {_fmt(parsed['final_top5'])}",
        "",
        "## Expert summary",
        "",
        "| Expert | Train samples | Test samples | Class coverage | Accuracy (%) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for expert in parsed["experts"]:
        lines.append(
            "| {k} | {train} | {test} | {coverage} | {accuracy} |".format(
                k=expert,
                train=parsed["train_counts"].get(expert, 0),
                test=parsed["test_counts"].get(expert, 0),
                coverage=parsed["class_coverage"].get(expert, "N/A"),
                accuracy=parsed["expert_accuracy"].get(expert, "N/A"),
            )
        )
    lines += [
        "",
        "## Result interpretation",
        "",
        "- TODO: Compare this run with its matched baseline.",
        "- TODO: Explain routing balance, class specialization, and accuracy changes.",
        "",
        "## Next experiment decision",
        "",
        "- TODO: Record the next run and the evidence supporting that decision.",
        "",
    ]
    (run_dir / "9_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.run_dir / "7_full.log"
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    parsed = parse_log(text)
    write_metrics(args.run_dir, parsed, args.exit_code)
    write_expert_csv(args.run_dir, parsed)
    write_coverage_csv(args.run_dir, parsed)
    write_graphs(args.run_dir, parsed)
    write_summary(args.run_dir, parsed, args.exit_code)


if __name__ == "__main__":
    main()
