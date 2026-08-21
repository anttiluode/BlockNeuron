from __future__ import annotations

"""Gate 0X2 — factor-separated receptor composition.

Gate 0X1 collapsed "small bag" into one character-GRU vector before the shared
block. It learned the seen conjunctions but failed systematic visual
recombination on held-out pairs.

Gate 0X2 removes that upstream bottleneck. Object identity and visual quality
arrive as two physically separate semantic receptor populations:

    object receptor -------\
                            > SAME recurrent BlockNeuron -> public visual state
    quality receptor ------/

The exact same ten Fashion-MNIST class/attribute combinations remain held out.
No held-out combination occurs on any training path. The receptor-only path gets
no direct pixels, class labels, or attribute labels; it must align to the
image-trained public state.

This gate asks whether keeping factors separate until the block changes the
compositional failure observed in X1. It is not yet an architectural-advantage
claim: an ordinary factorized/additive latent is the mandatory next attacker.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blockneuron.receptor_composition import (  # noqa: E402
    ReceptorCompositionConfig,
    ReceptorCompositionModel,
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
    save_grid,
    transform_tensor,
)


class ReceptorFashionPairs(Dataset):
    """Seen image pairs plus separate object/attribute receptor IDs."""

    def __init__(
        self,
        hf_dataset,
        prototype_bank: Tensor,
        *,
        shuffle_attribute_receptors: bool = False,
    ) -> None:
        self.ds = hf_dataset
        self.prototype_bank = prototype_bank
        self.shuffle_attribute_receptors = shuffle_attribute_receptors

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int):
        item = self.ds[index]
        label = int(item["label"])
        allowed = [
            attr
            for attr in range(len(ATTRIBUTES))
            if attr != HOLDOUT_ATTR[label]
        ]
        visual_attr = random.choice(allowed)

        image = item["image"].convert("L")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)
        x = transform_tensor(x, visual_attr)

        receptor_attr = visual_attr
        if self.shuffle_attribute_receptors:
            # Destroy quality meaning while keeping semantic pairs inside the same
            # 30-pair support. A held-out receptor conjunction is never leaked.
            choices = [attr for attr in allowed if attr != visual_attr]
            receptor_attr = random.choice(choices)

        prototype = self.prototype_bank[label, visual_attr]
        return x, label, visual_attr, receptor_attr, prototype


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def limit_dataset(ds, limit: int | None, seed: int):
    if limit is None or limit <= 0 or limit >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(limit))


def compute_losses(
    model: ReceptorCompositionModel,
    image: Tensor,
    label: Tensor,
    visual_attr: Tensor,
    receptor_attr: Tensor,
    prototype: Tensor,
    *,
    phase_mode: str,
) -> tuple[Tensor, dict[str, float]]:
    # Same physical block, three operating conditions.
    receptor_only = model.encode(
        object_ids=label,
        attribute_ids=receptor_attr,
        phase_mode=phase_mode,
    )
    image_only = model.encode(image=image, phase_mode=phase_mode)
    paired = model.encode(
        object_ids=label,
        attribute_ids=receptor_attr,
        image=image,
        phase_mode=phase_mode,
    )

    pub_r = receptor_only["public"]
    pub_i = image_only["public"]
    pub_p = paired["public"]
    det_r = receptor_only["detail"]
    det_i = image_only["detail"]
    det_p = paired["detail"]
    logits_i = image_only["logits"]
    attr_logits_i = image_only["attr_logits"]
    assert all(
        isinstance(v, Tensor)
        for v in [
            pub_r,
            pub_i,
            pub_p,
            det_r,
            det_i,
            det_p,
            logits_i,
            attr_logits_i,
        ]
    )

    # Pixel structure is learned from image evidence. The receptor-only path has
    # no pixel target.
    recon_i = model.decode(pub_i, det_i)
    recon_p = model.decode(pub_p, det_p)
    public_i = model.decode(pub_i, None)

    loss_recon = F.mse_loss(recon_i, image)
    loss_pair_recon = F.mse_loss(recon_p, image)
    loss_public = F.mse_loss(public_i, prototype)

    # Receptor-only state acquires visual meaning by agreeing with the image-side
    # public state for the paired experience.
    loss_align = (
        1.0 - F.cosine_similarity(pub_r, pub_i.detach(), dim=-1)
    ).mean()
    loss_pair_align = 0.5 * (
        (1.0 - F.cosine_similarity(pub_p, pub_i.detach(), dim=-1)).mean()
        + (1.0 - F.cosine_similarity(pub_p, pub_r.detach(), dim=-1)).mean()
    )

    # Class and quality probes are taught ONLY from image-derived public state.
    loss_class = F.cross_entropy(logits_i, label)
    loss_attr = F.cross_entropy(attr_logits_i, visual_attr)

    loss_receptor_detail = det_r.square().mean()
    loss_detail_size = 0.05 * det_i.square().mean()

    loss = (
        1.00 * loss_recon
        + 0.70 * loss_pair_recon
        + 0.80 * loss_public
        + 0.80 * loss_align
        + 0.25 * loss_pair_align
        + 0.35 * loss_class
        + 0.25 * loss_attr
        + 0.08 * loss_receptor_detail
        + loss_detail_size
    )

    return loss, {
        "loss": float(loss.detach()),
        "recon": float(loss_recon.detach()),
        "public": float(loss_public.detach()),
        "align": float(loss_align.detach()),
        "class": float(loss_class.detach()),
        "attr": float(loss_attr.detach()),
    }


@torch.no_grad()
def evaluate_compositions(
    model: ReceptorCompositionModel,
    prototype_bank: Tensor,
    *,
    device: torch.device,
    phase_mode: str,
) -> tuple[dict[str, float], Tensor, Tensor]:
    """Evaluate all 40 receptor pairs, including ten unseen conjunctions."""
    model.eval()

    pairs = [
        (label, attr)
        for label in range(len(LABELS))
        for attr in range(len(ATTRIBUTES))
    ]
    labels = torch.tensor([p[0] for p in pairs], device=device)
    attrs = torch.tensor([p[1] for p in pairs], device=device)

    out = model.encode(
        object_ids=labels,
        attribute_ids=attrs,
        phase_mode=phase_mode,
    )
    pub = out["public"]
    logits = out["logits"]
    attr_logits = out["attr_logits"]
    assert isinstance(pub, Tensor)
    assert isinstance(logits, Tensor)
    assert isinstance(attr_logits, Tensor)

    generated = model.decode(pub, None)

    held = torch.tensor(
        [p[1] == HOLDOUT_ATTR[p[0]] for p in pairs],
        device=device,
        dtype=torch.bool,
    )
    seen = ~held

    class_pred = logits.argmax(-1)
    attr_pred = attr_logits.argmax(-1)
    joint_ok = (class_pred == labels) & (attr_pred == attrs)

    targets = torch.stack(
        [prototype_bank[label, attr] for label, attr in pairs]
    ).to(device)
    per_item_mse = (generated - targets).square().flatten(1).mean(1)

    all_targets = prototype_bank.to(device).view(
        len(LABELS) * len(ATTRIBUTES), 1, 28, 28
    )
    nn_dist = (
        generated[:, None] - all_targets[None]
    ).square().flatten(2).mean(2)
    nearest = nn_dist.argmin(dim=1)
    nn_class = nearest // len(ATTRIBUTES)
    nn_attr = nearest % len(ATTRIBUTES)
    nn_joint = (nn_class == labels) & (nn_attr == attrs)

    def mean_mask(x: Tensor, mask: Tensor) -> float:
        return float(x[mask].float().mean()) if bool(mask.any()) else float("nan")

    metrics: dict[str, float] = {}
    for name, mask in [("seen", seen), ("heldout", held)]:
        metrics[f"{name}_class_acc"] = mean_mask(class_pred == labels, mask)
        metrics[f"{name}_attr_acc"] = mean_mask(attr_pred == attrs, mask)
        metrics[f"{name}_joint_acc"] = mean_mask(joint_ok, mask)
        metrics[f"{name}_proto_mse"] = mean_mask(per_item_mse, mask)
        metrics[f"{name}_visual_nn_class"] = mean_mask(nn_class == labels, mask)
        metrics[f"{name}_visual_nn_attr"] = mean_mask(nn_attr == attrs, mask)
        metrics[f"{name}_visual_nn_joint"] = mean_mask(nn_joint, mask)

    grid = generated.view(len(LABELS), len(ATTRIBUTES), 1, 28, 28)
    public_grid = pub.view(len(LABELS), len(ATTRIBUTES), -1)
    return metrics, grid, public_grid


def print_metrics(prefix: str, metrics: dict[str, float]) -> None:
    keys = [
        "seen_joint_acc",
        "heldout_joint_acc",
        "heldout_class_acc",
        "heldout_attr_acc",
        "seen_proto_mse",
        "heldout_proto_mse",
        "seen_visual_nn_joint",
        "heldout_visual_nn_joint",
    ]
    print(
        f"{prefix}: "
        + "  ".join(f"{key}={metrics[key]:.4f}" for key in keys)
    )


@torch.no_grad()
def decode_receptor_trajectory(
    model: ReceptorCompositionModel,
    pairs: list[tuple[int, int]],
    *,
    device: torch.device,
    phase_mode: str,
) -> Tensor:
    labels = torch.tensor([p[0] for p in pairs], device=device)
    attrs = torch.tensor([p[1] for p in pairs], device=device)
    out = model.encode(
        object_ids=labels,
        attribute_ids=attrs,
        phase_mode=phase_mode,
        return_trace=True,
    )
    trace = out["trace"]
    assert isinstance(trace, dict)
    states = trace["states"]
    frames = []
    for step in range(states.shape[1]):
        frames.append(model.decode_state(states[:, step], keep_detail=False))
    return torch.stack(frames, dim=1)


@torch.no_grad()
def write_reports(
    model: ReceptorCompositionModel,
    prototype_bank: Tensor,
    output_dir: Path,
    *,
    device: torch.device,
    phase_mode: str,
    scale: int,
) -> dict[str, float]:
    metrics, grid, _ = evaluate_compositions(
        model,
        prototype_bank,
        device=device,
        phase_mode=phase_mode,
    )

    held_mask = torch.zeros(len(LABELS), len(ATTRIBUTES), dtype=torch.bool)
    for label, attr in heldout_pairs():
        held_mask[label, attr] = True

    save_grid(
        grid,
        output_dir / "all_receptor_compositions.png",
        row_labels=LABELS,
        column_labels=ATTRIBUTES,
        heldout_mask=held_mask,
        scale=scale,
    )

    held_gen = torch.stack([grid[label, attr] for label, attr in heldout_pairs()])
    held_target = torch.stack(
        [prototype_bank[label, attr] for label, attr in heldout_pairs()]
    ).to(held_gen.device)
    compare = torch.stack([held_gen, held_target], dim=1)
    held_names = [canonical_prompt(label, attr) for label, attr in heldout_pairs()]
    save_grid(
        compare,
        output_dir / "heldout_compare.png",
        row_labels=held_names,
        column_labels=["generated", "target"],
        scale=scale,
    )

    trajectory = decode_receptor_trajectory(
        model,
        heldout_pairs(),
        device=device,
        phase_mode=phase_mode,
    )
    save_grid(
        trajectory,
        output_dir / "heldout_trajectory.png",
        row_labels=held_names,
        column_labels=[str(i + 1) for i in range(trajectory.shape[1])],
        scale=scale,
    )
    return metrics


def train_main(args: argparse.Namespace, device: torch.device) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Gate 0X2 needs Hugging Face datasets. Install with: "
            "python -m pip install -e '.[crossmodal]'"
        ) from exc

    assert_split_is_compositional()

    print(f"device={device}")
    print(f"dataset={DATASET_NAME}")
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

    config = ReceptorCompositionConfig(
        steps=args.steps,
        branches=args.branches,
        phase_dim=args.phase_dim,
        state_dim=args.state_dim,
        public_dim=args.public_dim,
        detail_dim=args.detail_dim,
        basis_dim=args.basis_dim,
        object_dim=args.object_dim,
        attribute_dim=args.attribute_dim,
    )
    model = ReceptorCompositionModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"train rows={len(train_ds)}")
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}")
    print(
        f"receptor dims: object={config.object_dim} "
        f"attribute={config.attribute_dim}"
    )
    print("attributes:", ", ".join(ATTRIBUTES))
    print("held-out combinations:")
    for label, attr in heldout_pairs():
        print(f"  {canonical_prompt(label, attr)}")
    print(
        "anti-cheat: object and attribute remain separate until the BlockNeuron; "
        "held-out receptor pairs never occur in training; receptor-only state "
        "receives no direct pixel/class/attribute target"
    )
    if args.shuffle_attribute_receptors:
        print(
            "CONTROL ACTIVE: quality receptor IDs are deliberately mismatched "
            "to image transformations"
        )

    best_seen_score = -1e9
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
                phase_mode=args.phase_mode,
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

        train_metrics = {
            key: value / max(batches, 1) for key, value in sums.items()
        }
        eval_metrics, _, _ = evaluate_compositions(
            model,
            prototype_bank,
            device=device,
            phase_mode=args.phase_mode,
        )

        print(
            f"epoch {epoch:02d} train: "
            + "  ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
        )
        print_metrics(f"epoch {epoch:02d} receptor composition", eval_metrics)

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
        }
        torch.save(checkpoint, output_dir / "last.pt")

        # Never select a checkpoint using held-out performance.
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
                phase_mode=args.phase_mode,
                scale=args.scale,
            )

    final_metrics = write_reports(
        model,
        prototype_bank,
        output_dir,
        device=device,
        phase_mode=args.phase_mode,
        scale=args.scale,
    )
    print("\nGate 0X2 receptor-composition training complete")
    print_metrics("final", final_metrics)
    print(f"checkpoint: {output_dir / 'best.pt'}")
    print(
        f"all compositions: "
        f"{output_dir / 'all_receptor_compositions.png'}"
    )
    print(f"held-out comparison: {output_dir / 'heldout_compare.png'}")
    print(f"held-out trajectory: {output_dir / 'heldout_trajectory.png'}")
    print(
        "Interpretation: compare heldout_* to Gate 0X1. If factor-separated "
        "receptors rescue unseen visual composition, the upstream merged-text "
        "bottleneck mattered. A factorized ordinary latent remains the "
        "mandatory attacker."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--train-limit", type=int, default=12_000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=18_001)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="default is control-aware so shuffled runs cannot overwrite baseline",
    )
    parser.add_argument(
        "--phase-mode", choices=["dynamic", "clamped"], default="dynamic"
    )
    parser.add_argument(
        "--shuffle-attribute-receptors",
        action="store_true",
        help="negative control: mismatch quality receptor ID from visual transform",
    )
    parser.add_argument(
        "--max-batches", type=int, default=0, help="debug: cap training batches per epoch"
    )
    parser.add_argument("--report-every", type=int, default=4)
    parser.add_argument("--scale", type=int, default=4)

    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--phase-dim", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--public-dim", type=int, default=24)
    parser.add_argument("--detail-dim", type=int, default=32)
    parser.add_argument("--basis-dim", type=int, default=128)
    parser.add_argument("--object-dim", type=int, default=16)
    parser.add_argument("--attribute-dim", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = (
            "runs/gate0x2_receptor_attr_shuffled"
            if args.shuffle_attribute_receptors
            else "runs/gate0x2_receptor"
        )
    seed_everything(args.seed)
    train_main(args, choose_device(args.device))


if __name__ == "__main__":
    main()
