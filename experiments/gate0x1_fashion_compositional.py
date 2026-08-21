from __future__ import annotations

"""Gate 0X1 — compositional text/image completion with held-out combinations.

This keeps the Gate 0X0 anti-cheat and makes the task harder:

* Start from Fashion-MNIST.
* Add controlled, reusable visual attributes: small, large, left, right.
* Hold out exactly one class/attribute combination per class.
* The model sees every class and every attribute during training, but NEVER the
  ten held-out combinations.
* Text receives no pixel, class, or attribute supervision.
* Image-derived public state is taught class and attribute structure.
* At test time, text-only queries for unseen combinations must compose the two.

Example:
    training may contain "small sneaker" and "right bag"
    but never "small bag"

Then query:
    "small bag"

If the text-only public state and visual readout are correct, the system has
reused learned semantic/visual factors rather than only memorizing 40 phrases.

This is a compositional mechanism test, not a novelty or engineering-advantage claim.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Allow `python experiments/...py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blockneuron.crossmodal import (  # noqa: E402
    CharTokenizer,
    CrossModalBlockModel,
    CrossModalConfig,
)


DATASET_NAME = "anonyme449/fashion_mnist"
LABELS = [
    "t-shirt top",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
]
ATTRIBUTES = ["small", "large", "left", "right"]

# One withheld attribute per class. Each attribute is withheld from multiple
# classes, while remaining visible with other classes.
HOLDOUT_ATTR = {label: label % len(ATTRIBUTES) for label in range(len(LABELS))}

PROMPT_TEMPLATES = [
    "{attr} {label}",
    "a {attr} {label}",
    "an image of a {attr} {label}",
    "{label} that is {attr}",
    "show me a {attr} {label}",
]


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


def transform_tensor(image: Tensor, attr: int | str) -> Tensor:
    """Apply one controlled 28x28 transformation without torchvision."""
    if isinstance(attr, str):
        attr = ATTRIBUTES.index(attr)
    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError("image must be [1,H,W]")
    _, h, w = image.shape
    if h != w:
        raise ValueError("expected square image")

    if attr == 0:  # small
        side = int(round(h * 0.72))
        small = F.interpolate(
            image[None], size=(side, side), mode="bilinear", align_corners=False
        )[0]
        out = torch.zeros_like(image)
        top = (h - side) // 2
        left = (w - side) // 2
        out[:, top : top + side, left : left + side] = small
        return out

    if attr == 1:  # large
        side = int(round(h * 1.22))
        large = F.interpolate(
            image[None], size=(side, side), mode="bilinear", align_corners=False
        )[0]
        top = (side - h) // 2
        left = (side - w) // 2
        return large[:, top : top + h, left : left + w]

    shift = 4
    out = torch.zeros_like(image)
    if attr == 2:  # left
        out[:, :, : w - shift] = image[:, :, shift:]
        return out
    if attr == 3:  # right
        out[:, :, shift:] = image[:, :, : w - shift]
        return out
    raise ValueError(f"unknown attribute index {attr}")


def canonical_prompt(label: int, attr: int) -> str:
    return f"{ATTRIBUTES[attr]} {LABELS[label]}"


def random_prompt(label: int, attr: int) -> str:
    template = random.choice(PROMPT_TEMPLATES)
    return template.format(label=LABELS[label], attr=ATTRIBUTES[attr])


def heldout_pairs() -> list[tuple[int, int]]:
    return [(label, HOLDOUT_ATTR[label]) for label in range(len(LABELS))]


def seen_pairs() -> list[tuple[int, int]]:
    return [
        (label, attr)
        for label in range(len(LABELS))
        for attr in range(len(ATTRIBUTES))
        if attr != HOLDOUT_ATTR[label]
    ]


def assert_split_is_compositional() -> None:
    held = set(heldout_pairs())
    seen = set(seen_pairs())
    assert not (held & seen)
    assert len(held) == len(LABELS)
    assert len(seen) == len(LABELS) * (len(ATTRIBUTES) - 1)
    for label in range(len(LABELS)):
        assert any(pair[0] == label for pair in seen)
    for attr in range(len(ATTRIBUTES)):
        assert any(pair[1] == attr for pair in seen)
        assert any(pair[1] == attr for pair in held)


@torch.no_grad()
def compute_class_means(hf_dataset) -> Tensor:
    sums = torch.zeros(len(LABELS), 1, 28, 28)
    counts = torch.zeros(len(LABELS))
    for item in hf_dataset:
        image = item["image"].convert("L")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)
        label = int(item["label"])
        sums[label] += x
        counts[label] += 1
    if (counts == 0).any():
        raise RuntimeError("dataset subset omitted at least one Fashion-MNIST class")
    return sums / counts[:, None, None, None]


@torch.no_grad()
def build_prototype_bank(class_means: Tensor) -> Tensor:
    """[classes, attrs, 1, 28, 28] deterministic visual-family targets."""
    rows = []
    for label in range(len(LABELS)):
        rows.append(
            torch.stack(
                [transform_tensor(class_means[label], attr) for attr in range(len(ATTRIBUTES))]
            )
        )
    return torch.stack(rows)


class CompositionalFashionPairs(Dataset):
    """Training set: sample only SEEN class/attribute combinations."""

    def __init__(
        self,
        hf_dataset,
        prototype_bank: Tensor,
        *,
        shuffle_attribute_words: bool = False,
    ) -> None:
        self.ds = hf_dataset
        self.prototype_bank = prototype_bank
        self.shuffle_attribute_words = shuffle_attribute_words

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int):
        item = self.ds[index]
        label = int(item["label"])
        allowed = [
            attr for attr in range(len(ATTRIBUTES)) if attr != HOLDOUT_ATTR[label]
        ]
        attr = random.choice(allowed)

        image = item["image"].convert("L")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)
        x = transform_tensor(x, attr)

        text_attr = attr
        if self.shuffle_attribute_words:
            choices = [a for a in range(len(ATTRIBUTES)) if a != attr]
            text_attr = random.choice(choices)
        text = random_prompt(label, text_attr)

        prototype = self.prototype_bank[label, attr]
        return x, label, attr, text, prototype


def save_grid(
    images: Tensor,
    path: Path,
    *,
    row_labels: list[str] | None = None,
    column_labels: list[str] | None = None,
    heldout_mask: Tensor | None = None,
    scale: int = 4,
) -> None:
    """Save [rows, cols, 1, H, W] as a labelled grayscale sheet."""
    x = images.detach().float().cpu().clamp(0, 1)
    if x.ndim == 4:
        x = x[:, None]
    if x.ndim != 5:
        raise ValueError("images must be [rows,cols,1,H,W] or [rows,1,H,W]")
    rows, cols, _, h, w = x.shape
    label_w = 110 if row_labels else 0
    header_h = 20 if column_labels else 4
    canvas = Image.new(
        "L", (label_w + cols * w * scale, header_h + rows * h * scale), 255
    )
    draw = ImageDraw.Draw(canvas)

    if column_labels:
        for c, name in enumerate(column_labels):
            draw.text((label_w + c * w * scale + 2, 3), name, fill=0)

    for r in range(rows):
        y0 = header_h + r * h * scale
        if row_labels:
            draw.text((2, y0 + 4), row_labels[r], fill=0)
        for c in range(cols):
            x0 = label_w + c * w * scale
            arr = (x[r, c, 0].numpy() * 255.0).astype(np.uint8)
            tile = Image.fromarray(arr, mode="L").resize(
                (w * scale, h * scale), Image.Resampling.NEAREST
            )
            canvas.paste(tile, (x0, y0))
            if heldout_mask is not None and bool(heldout_mask[r, c]):
                draw.rectangle((x0 + 2, y0 + 2, x0 + 12, y0 + 12), outline=0, width=1)
                draw.text((x0 + 4, y0 + 2), "H", fill=0)
    canvas.save(path)


def decode_text_trajectory(
    model: CrossModalBlockModel,
    tokenizer: CharTokenizer,
    texts: list[str],
    *,
    device: torch.device,
    phase_mode: str,
) -> Tensor:
    tokens = tokenizer.encode(texts, device=device)
    encoded = model.encode(tokens=tokens, phase_mode=phase_mode, return_trace=True)
    trace = encoded["trace"]
    assert isinstance(trace, dict)
    states = trace["states"]
    frames = []
    for step in range(states.shape[1]):
        frames.append(model.decode_state(states[:, step], keep_detail=False))
    return torch.stack(frames, dim=1)


def compute_losses(
    model: CrossModalBlockModel,
    attr_head: nn.Linear,
    tokenizer: CharTokenizer,
    image: Tensor,
    label: Tensor,
    attr: Tensor,
    texts: list[str],
    prototype: Tensor,
    *,
    phase_mode: str,
) -> tuple[Tensor, dict[str, float]]:
    tokens = tokenizer.encode(texts, device=image.device)

    text_only = model.encode(tokens=tokens, phase_mode=phase_mode)
    image_only = model.encode(image=image, phase_mode=phase_mode)
    paired = model.encode(tokens=tokens, image=image, phase_mode=phase_mode)

    pub_t = text_only["public"]
    pub_i = image_only["public"]
    pub_p = paired["public"]
    det_t = text_only["detail"]
    det_i = image_only["detail"]
    det_p = paired["detail"]
    logits_i = image_only["logits"]
    logits_p = paired["logits"]
    assert all(
        isinstance(v, Tensor)
        for v in [pub_t, pub_i, pub_p, det_t, det_i, det_p, logits_i, logits_p]
    )

    recon_i = model.decode(pub_i, det_i)
    recon_p = model.decode(pub_p, det_p)
    public_i = model.decode(pub_i, None)

    # Exact instance information belongs to public+private.
    loss_recon = F.mse_loss(recon_i, image)
    loss_pair_recon = F.mse_loss(recon_p, image)

    # Public state is explicitly taught a stable CLASS + ATTRIBUTE visual family.
    # This target comes ONLY from image-side training statistics.
    loss_public = F.mse_loss(public_i, prototype)

    # Text gets no pixels and no class/attribute labels: it can only align to the
    # public state produced by its paired image.
    loss_align = (1.0 - F.cosine_similarity(pub_t, pub_i.detach(), dim=-1)).mean()
    loss_pair_align = 0.5 * (
        (1.0 - F.cosine_similarity(pub_p, pub_i.detach(), dim=-1)).mean()
        + (1.0 - F.cosine_similarity(pub_p, pub_t.detach(), dim=-1)).mean()
    )

    # Public geometry is supervised from IMAGE evidence only.
    loss_class = F.cross_entropy(logits_i, label)
    loss_pair_class = F.cross_entropy(logits_p, label)
    loss_attr = F.cross_entropy(attr_head(pub_i), attr)

    loss_text_detail = det_t.square().mean()
    loss_detail_size = 0.05 * det_i.square().mean()

    loss = (
        1.00 * loss_recon
        + 0.70 * loss_pair_recon
        + 0.80 * loss_public
        + 0.80 * loss_align
        + 0.25 * loss_pair_align
        + 0.35 * loss_class
        + 0.10 * loss_pair_class
        + 0.25 * loss_attr
        + 0.08 * loss_text_detail
        + loss_detail_size
    )

    metrics = {
        "loss": float(loss.detach()),
        "recon": float(loss_recon.detach()),
        "public": float(loss_public.detach()),
        "align": float(loss_align.detach()),
        "class": float(loss_class.detach()),
        "attr": float(loss_attr.detach()),
    }
    return loss, metrics


@torch.no_grad()
def evaluate_compositions(
    model: CrossModalBlockModel,
    attr_head: nn.Linear,
    tokenizer: CharTokenizer,
    prototype_bank: Tensor,
    *,
    device: torch.device,
    phase_mode: str,
) -> tuple[dict[str, float], Tensor]:
    """Evaluate all 40 text-only combinations, including ten never trained."""
    model.eval()
    attr_head.eval()

    pairs = [
        (label, attr)
        for label in range(len(LABELS))
        for attr in range(len(ATTRIBUTES))
    ]
    texts = [canonical_prompt(label, attr) for label, attr in pairs]
    tokens = tokenizer.encode(texts, device=device)
    out = model.encode(tokens=tokens, phase_mode=phase_mode)
    pub = out["public"]
    logits = out["logits"]
    assert isinstance(pub, Tensor) and isinstance(logits, Tensor)

    generated = model.decode(pub, None)
    attr_logits = attr_head(pub)

    labels = torch.tensor([p[0] for p in pairs], device=device)
    attrs = torch.tensor([p[1] for p in pairs], device=device)
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

    return metrics, generated.view(len(LABELS), len(ATTRIBUTES), 1, 28, 28)


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
    body = "  ".join(f"{k}={metrics[k]:.4f}" for k in keys)
    print(f"{prefix}: {body}")


@torch.no_grad()
def write_reports(
    model: CrossModalBlockModel,
    attr_head: nn.Linear,
    tokenizer: CharTokenizer,
    prototype_bank: Tensor,
    output_dir: Path,
    *,
    device: torch.device,
    phase_mode: str,
    scale: int,
) -> dict[str, float]:
    metrics, grid = evaluate_compositions(
        model,
        attr_head,
        tokenizer,
        prototype_bank,
        device=device,
        phase_mode=phase_mode,
    )

    held_mask = torch.zeros(len(LABELS), len(ATTRIBUTES), dtype=torch.bool)
    for label, attr in heldout_pairs():
        held_mask[label, attr] = True
    save_grid(
        grid,
        output_dir / "all_compositions.png",
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
    held_text = [
        canonical_prompt(label, attr) for label, attr in heldout_pairs()
    ]
    save_grid(
        compare,
        output_dir / "heldout_compare.png",
        row_labels=held_text,
        column_labels=["generated", "target"],
        scale=scale,
    )

    trajectory = decode_text_trajectory(
        model,
        tokenizer,
        held_text,
        device=device,
        phase_mode=phase_mode,
    )
    save_grid(
        trajectory,
        output_dir / "heldout_trajectory.png",
        row_labels=held_text,
        column_labels=[str(i + 1) for i in range(trajectory.shape[1])],
        scale=scale,
    )
    return metrics


def train_main(args: argparse.Namespace, device: torch.device) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Gate 0X1 needs Hugging Face datasets. "
            "Install with: python -m pip install -e '.[crossmodal]'"
        ) from exc

    assert_split_is_compositional()
    print(f"device={device}")
    print(f"dataset={DATASET_NAME}")
    print("loading Hugging Face Fashion-MNIST...")

    train_hf = load_dataset(DATASET_NAME, split="train")
    train_hf = limit_dataset(train_hf, args.train_limit, args.seed)

    # Only training images define the visual-family prototypes.
    class_means = compute_class_means(train_hf)
    prototype_bank = build_prototype_bank(class_means)

    train_ds = CompositionalFashionPairs(
        train_hf,
        prototype_bank,
        shuffle_attribute_words=args.shuffle_attribute_words,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    config = CrossModalConfig(
        steps=args.steps,
        branches=args.branches,
        phase_dim=args.phase_dim,
        state_dim=args.state_dim,
        public_dim=args.public_dim,
        detail_dim=args.detail_dim,
        basis_dim=args.basis_dim,
    )
    tokenizer = CharTokenizer(max_len=config.max_text_len)
    model = CrossModalBlockModel(config).to(device)
    attr_head = nn.Linear(config.public_dim, len(ATTRIBUTES)).to(device)

    parameters = list(model.parameters()) + list(attr_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"train rows={len(train_ds)}")
    print(
        f"parameters={sum(p.numel() for p in parameters):,} "
        f"(model={sum(p.numel() for p in model.parameters()):,})"
    )
    print("attributes:", ", ".join(ATTRIBUTES))
    print("held-out combinations:")
    for label, attr in heldout_pairs():
        print(f"  {canonical_prompt(label, attr)}")
    print(
        "anti-cheat: held-out combinations never occur in training; "
        "the text-only path receives no pixel/class/attribute target"
    )
    if args.shuffle_attribute_words:
        print(
            "CONTROL ACTIVE: attribute words are deliberately mismatched "
            "to image transformations"
        )

    best_seen_score = -1e9
    for epoch in range(1, args.epochs + 1):
        model.train()
        attr_head.train()
        sums: dict[str, float] = {}
        batches = 0

        for image, label, attr, texts, prototype in train_loader:
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            attr = attr.to(device, non_blocking=True)
            prototype = prototype.to(device, non_blocking=True)
            texts = list(texts)

            loss, batch_metrics = compute_losses(
                model,
                attr_head,
                tokenizer,
                image,
                label,
                attr,
                texts,
                prototype,
                phase_mode=args.phase_mode,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 2.0)
            optimizer.step()

            for k, v in batch_metrics.items():
                sums[k] = sums.get(k, 0.0) + v
            batches += 1
            if args.max_batches and batches >= args.max_batches:
                break

        train_metrics = {k: v / max(batches, 1) for k, v in sums.items()}
        eval_metrics, _ = evaluate_compositions(
            model,
            attr_head,
            tokenizer,
            prototype_bank,
            device=device,
            phase_mode=args.phase_mode,
        )

        body = "  ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
        print(f"epoch {epoch:02d} train: {body}")
        print_metrics(f"epoch {epoch:02d} composition", eval_metrics)

        checkpoint = {
            "model": model.state_dict(),
            "attr_head": attr_head.state_dict(),
            "config": asdict(config),
            "alphabet": tokenizer.alphabet,
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

        # IMPORTANT: do not choose a checkpoint using held-out performance.
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
                attr_head,
                tokenizer,
                prototype_bank,
                output_dir,
                device=device,
                phase_mode=args.phase_mode,
                scale=args.scale,
            )

    final_metrics = write_reports(
        model,
        attr_head,
        tokenizer,
        prototype_bank,
        output_dir,
        device=device,
        phase_mode=args.phase_mode,
        scale=args.scale,
    )

    print("\nGate 0X1 compositional training complete")
    print_metrics("final", final_metrics)
    print(f"checkpoint: {output_dir / 'best.pt'}")
    print(f"all compositions: {output_dir / 'all_compositions.png'}")
    print(f"held-out comparison: {output_dir / 'heldout_compare.png'}")
    print(f"held-out trajectory: {output_dir / 'heldout_trajectory.png'}")
    print(
        "Interpretation: heldout_* metrics are the real gate. "
        "Seen success alone can still be phrase memorization."
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
    parser.add_argument("--output-dir", default="runs/gate0x1_compositional")
    parser.add_argument(
        "--phase-mode", choices=["dynamic", "clamped"], default="dynamic"
    )
    parser.add_argument(
        "--shuffle-attribute-words",
        action="store_true",
        help="negative control: mismatch attribute word from visual transform",
    )
    parser.add_argument(
        "--max-batches", type=int, default=0, help="debug: cap batches per epoch"
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    train_main(args, device)


if __name__ == "__main__":
    main()
