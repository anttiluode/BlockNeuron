from __future__ import annotations

"""Run Gate 0X2 across seeds beside matched ordinary factorized attackers.

Default workload: 5 seeds x {X2, MLP, GRU}. Each run keeps the same 12k-image,
16-epoch held-out-composition protocol and writes its own directory. The suite
then reads final/seen-selected checkpoints and writes CSV/JSON/Markdown summaries.

This is intentionally a launcher/aggregator rather than another learning model.
"""

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Iterable

import torch

DEFAULT_SEEDS = [18_001, 18_002, 18_003, 18_004, 18_005]
DEFAULT_MODELS = ["x2", "mlp", "gru"]
METRICS = [
    "seen_joint_acc",
    "heldout_joint_acc",
    "heldout_class_acc",
    "heldout_attr_acc",
    "seen_proto_mse",
    "heldout_proto_mse",
    "seen_visual_nn_joint",
    "heldout_visual_nn_joint",
]


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_str_list(text: str) -> list[str]:
    values = [part.strip().lower() for part in text.split(",") if part.strip()]
    unknown = sorted(set(values) - set(DEFAULT_MODELS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown models: {', '.join(unknown)}")
    return values


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def run_one(
    *,
    model: str,
    seed: int,
    root: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    out_dir = root / model / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if model == "x2":
        cmd = [
            sys.executable,
            "experiments/gate0x2_receptor_composition.py",
        ]
        reference_params = 131_320
    else:
        cmd = [
            sys.executable,
            "experiments/gate0x2_factorized_attacker.py",
            "--attacker",
            model,
        ]
        reference_params = None

    cmd += [
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--train-limit",
        str(args.train_limit),
        "--batch-size",
        str(args.batch_size),
        "--output-dir",
        str(out_dir),
        "--report-every",
        str(args.report_every),
        "--device",
        args.device,
    ]
    if args.max_batches:
        cmd += ["--max-batches", str(args.max_batches)]

    print("\n" + "=" * 78)
    print(f"RUN model={model} seed={seed}")
    print(" ".join(cmd))
    print("=" * 78)
    subprocess.run(cmd, check=True)

    last = load_checkpoint(out_dir / "last.pt")
    best = load_checkpoint(out_dir / "best.pt")
    final_eval = last["eval"]
    selected_eval = best["eval"]
    params = last.get("parameters", reference_params)

    row: dict[str, object] = {
        "model": model,
        "seed": seed,
        "parameters": params,
        "final_epoch": int(last["epoch"]),
        "selected_epoch": int(best["epoch"]),
    }
    for metric in METRICS:
        row[f"final_{metric}"] = float(final_eval[metric])
        row[f"selected_{metric}"] = float(selected_eval[metric])
    return row


def mean_std(rows: Iterable[dict[str, object]], key: str) -> tuple[float, float]:
    values = [float(row[key]) for row in rows]
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def aggregate(rows: list[dict[str, object]]) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for model in DEFAULT_MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        if not model_rows:
            continue
        summary[model] = {}
        for prefix in ["final", "selected"]:
            summary[model][prefix] = {}
            for metric in METRICS:
                mean, std = mean_std(model_rows, f"{prefix}_{metric}")
                summary[model][prefix][f"{metric}_mean"] = mean
                summary[model][prefix][f"{metric}_std"] = std
        params = [row["parameters"] for row in model_rows if row["parameters"] is not None]
        if params:
            summary[model]["parameters"] = {
                "mean": statistics.mean(float(v) for v in params),
                "std": statistics.stdev(float(v) for v in params) if len(params) > 1 else 0.0,
            }
    return summary


def write_markdown(
    path: Path,
    rows: list[dict[str, object]],
    summary: dict[str, dict[str, dict[str, float]]],
) -> None:
    lines = [
        "# Gate 0X2 replication + ordinary attackers",
        "",
        "The table below reports **final-epoch mean ± sample SD across seeds**.",
        "Checkpoint selection in the underlying runs never uses held-out metrics.",
        "",
        "| model | params | seen joint | held-out joint | held-out attr | held-out proto MSE | seen visual NN | held-out visual NN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in DEFAULT_MODELS:
        if model not in summary:
            continue
        s = summary[model]["final"]
        p = summary[model].get("parameters", {}).get("mean", float("nan"))

        def fmt(metric: str) -> str:
            return f"{s[metric + '_mean']:.4f} ± {s[metric + '_std']:.4f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    f"{p:,.0f}",
                    fmt("seen_joint_acc"),
                    fmt("heldout_joint_acc"),
                    fmt("heldout_attr_acc"),
                    fmt("heldout_proto_mse"),
                    fmt("seen_visual_nn_joint"),
                    fmt("heldout_visual_nn_joint"),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Per-seed held-out receipt",
        "",
        "| model | seed | selected epoch | final held-out MSE | final held-out visual NN | selected held-out visual NN |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['seed']} | {row['selected_epoch']} | "
            f"{float(row['final_heldout_proto_mse']):.4f} | "
            f"{float(row['final_heldout_visual_nn_joint']):.4f} | "
            f"{float(row['selected_heldout_visual_nn_joint']):.4f} |"
        )

    lines += [
        "",
        "## Interpretation rule",
        "",
        "- If MLP/GRU match X2 across seeds, **factor separation** explains the X2 rescue.",
        "- If X2 retains a reproducible held-out advantage at comparable parameter count, the structured block earns the next mechanistic attack.",
        "- Do not choose seeds, epochs, or checkpoints by held-out performance.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--output-dir", default="runs/gate0x2_replication")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--train-limit", type=int, default=12_000)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--report-every", type=int, default=16)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = parse_int_list(args.seeds)
    models = parse_str_list(args.models)
    if not seeds or not models:
        raise SystemExit("at least one seed and one model are required")

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for model in models:
            rows.append(run_one(model=model, seed=seed, root=root, args=args))

    fieldnames = list(rows[0].keys())
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = aggregate(rows)
    (root / "summary.json").write_text(
        json.dumps({"runs": rows, "aggregate": summary}, indent=2),
        encoding="utf-8",
    )
    write_markdown(root / "SUMMARY.md", rows, summary)

    print("\nReplication suite complete")
    print(f"CSV:      {root / 'summary.csv'}")
    print(f"JSON:     {root / 'summary.json'}")
    print(f"Markdown: {root / 'SUMMARY.md'}")
    for model in models:
        s = summary[model]["final"]
        print(
            f"{model:>3s}: heldout_mse={s['heldout_proto_mse_mean']:.4f}±{s['heldout_proto_mse_std']:.4f}  "
            f"heldout_visual={s['heldout_visual_nn_joint_mean']:.4f}±{s['heldout_visual_nn_joint_std']:.4f}"
        )


if __name__ == "__main__":
    main()
