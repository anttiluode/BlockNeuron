from __future__ import annotations

"""Gate 0X2 mandatory ordinary attackers.

The object and attribute factors remain separate, but all BlockNeuron machinery
is removed. Two baselines are supplied:

- ``mlp``: concatenate factorized object/attribute/image drives and fuse by MLP.
- ``gru``: present object, attribute, image as three ordinary tokens to a GRU.

Both use the same Fashion-MNIST split, image encoder, public/private heads,
coordinate decoder, losses, and held-out metrics as Gate 0X2. The receptor-only
path still receives no direct pixel/class/attribute target.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blockneuron.factorized_attacker import (  # noqa: E402
    FactorizedAttackerConfig,
    FactorizedAttackerModel,
)
from experiments.gate0x1_fashion_compositional import (  # noqa: E402
    ATTRIBUTES,
    DATASET_NAME,
    HOLDOUT_ATTR,
    LABELS,
    assert_split_is_compositional,
    build_prototype_bank,
    canonical_prompt,
    compute_class_means,
    heldout_pairs,
)
from experiments.gate0x2_receptor_composition import (  # noqa: E402
    ReceptorFashionPairs,
    choose_device,
    compute_losses,
    evaluate_compositions,
    limit_dataset,
    print_metrics,
    seed_everything,
    write_reports,
)

X2_REFERENCE_PARAMS = 131_320


def train_main(args: argparse.Namespace, device: torch.device) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "This experiment needs Hugging Face datasets. Install with: "
            "python -m pip install -e '.[crossmodal]'"
        ) from exc

    assert_split_is_compositional()
    print(f"device={device}")
    print(f"dataset={DATASET_NAME}")
    print(f"attacker={args.attacker}")
    print("loading Hugging Face Fashion-MNIST...")
    train_hf = load_dataset(DATASET_NAME, split="train")
    train_hf = limit_dataset(train_hf, args.train_limit, args.seed)

    class_means = compute_class_means(train_hf)
    prototype_bank = build_prototype_bank(class_means)
    train_ds = ReceptorFashionPairs(
        train_hf,
        prototype_bank,
        shuffle_attribute_receptors=args.shuffle_attribute_receptors,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    config = FactorizedAttackerConfig(
        attacker=args.attacker,
        state_dim=args.state_dim,
        public_dim=args.public_dim,
        detail_dim=args.detail_dim,
        basis_dim=args.basis_dim,
        object_dim=args.object_dim,
        attribute_dim=args.attribute_dim,
        mlp_hidden=args.mlp_hidden,
        gru_token_dim=args.gru_token_dim,
        gru_hidden=args.gru_hidden,
    )
    model = FactorizedAttackerModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = sum(p.numel() for p in model.parameters())
    print(f"train rows={len(train_ds)}")
    print(f"parameters={params:,}  X2_reference={X2_REFERENCE_PARAMS:,}  ratio={params / X2_REFERENCE_PARAMS:.3f}")
    print("ordinary attacker: factorized inputs, no branch gates, no phase, no BlockNeuron recurrence")
    print("held-out combinations:")
    for label, attr in heldout_pairs():
        print(f"  {canonical_prompt(label, attr)}")
    print(
        "anti-cheat: held-out factor pairs never occur in training; semantic-only "
        "state receives no direct pixel/class/attribute target"
    )
    if args.shuffle_attribute_receptors:
        print("CONTROL ACTIVE: quality IDs are deliberately mismatched to image transformations")

    best_seen_score = -1e9
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums: dict[str, float] = {}
        batches = 0
        for image, label, visual_attr, receptor_attr, prototype in train_loader:
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            visual_attr = visual_attr.to(device, non_blocking=True)
            receptor_attr = receptor_attr.to(device, non_blocking=True)
            prototype = prototype.to(device, non_blocking=True)

            loss, batch_metrics = compute_losses(
                model,
                image,
                label,
                visual_attr,
                receptor_attr,
                prototype,
                phase_mode="dynamic",
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            for key, value in batch_metrics.items():
                sums[key] = sums.get(key, 0.0) + value
            batches += 1
            if args.max_batches and batches >= args.max_batches:
                break

        train_metrics = {key: value / max(batches, 1) for key, value in sums.items()}
        eval_metrics, _, _ = evaluate_compositions(
            model,
            prototype_bank,
            device=device,
            phase_mode="dynamic",
        )
        epoch_record = {"epoch": float(epoch), **eval_metrics}
        history.append(epoch_record)
        print(
            f"epoch {epoch:02d} train: "
            + "  ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
        )
        print_metrics(f"epoch {epoch:02d} {args.attacker} composition", eval_metrics)

        checkpoint = {
            "model": model.state_dict(),
            "config": asdict(config),
            "labels": LABELS,
            "attributes": ATTRIBUTES,
            "holdout_attr": HOLDOUT_ATTR,
            "prototype_bank": prototype_bank,
            "dataset": DATASET_NAME,
            "epoch": epoch,
            "args": vars(args),
            "eval": eval_metrics,
            "history": history,
            "parameters": params,
            "x2_reference_parameters": X2_REFERENCE_PARAMS,
        }
        torch.save(checkpoint, output_dir / "last.pt")

        # Selection remains strictly on seen combinations.
        seen_score = (
            eval_metrics["seen_joint_acc"]
            + eval_metrics["seen_visual_nn_joint"]
            - eval_metrics["seen_proto_mse"]
        )
        if seen_score > best_seen_score:
            best_seen_score = seen_score
            torch.save(checkpoint, output_dir / "best.pt")

        if epoch == args.epochs or epoch % args.report_every == 0:
            write_reports(
                model,
                prototype_bank,
                output_dir,
                device=device,
                phase_mode="dynamic",
                scale=args.scale,
            )

    final_metrics = write_reports(
        model,
        prototype_bank,
        output_dir,
        device=device,
        phase_mode="dynamic",
        scale=args.scale,
    )
    peak_visual = max(row["heldout_visual_nn_joint"] for row in history)
    print(f"\nGate 0X2 {args.attacker} attacker training complete")
    print_metrics("final", final_metrics)
    print(f"peak_heldout_visual_nn_joint={peak_visual:.4f}")
    print(f"checkpoint: {output_dir / 'best.pt'}")
    print(f"all compositions: {output_dir / 'all_receptor_compositions.png'}")
    print(f"held-out comparison: {output_dir / 'heldout_compare.png'}")
    print(
        "Interpretation: if this ordinary factorized attacker matches X2 across "
        "seeds, factor separation explains the rescue. If X2 remains better at "
        "matched budget, the structured block deserves further attack."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attacker", choices=["mlp", "gru"], default="mlp")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--train-limit", type=int, default=12_000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=18_001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--shuffle-attribute-receptors", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--report-every", type=int, default=4)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--public-dim", type=int, default=24)
    parser.add_argument("--detail-dim", type=int, default=32)
    parser.add_argument("--basis-dim", type=int, default=128)
    parser.add_argument("--object-dim", type=int, default=16)
    parser.add_argument("--attribute-dim", type=int, default=8)
    parser.add_argument("--mlp-hidden", type=int, default=512)
    parser.add_argument("--gru-token-dim", type=int, default=64)
    parser.add_argument("--gru-hidden", type=int, default=112)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        suffix = "_attr_shuffled" if args.shuffle_attribute_receptors else ""
        args.output_dir = f"runs/gate0x2_attacker_{args.attacker}{suffix}"
    seed_everything(args.seed)
    train_main(args, choose_device(args.device))


if __name__ == "__main__":
    main()
