from __future__ import annotations

"""Gate 0X executable — text/image completion through one shared BlockNeuron.

The first version is deliberately small and falsifiable:

* Dataset: Hugging Face Fashion-MNIST (28x28 grayscale, ten classes).
* Text: raw class strings encoded by a tiny character GRU; no pretrained CLIP.
* Image: tiny CNN receptor.
* Shared substrate: one recurrent multi-branch BlockNeuron with a T^K phase path.
* Readout: a public concept state + private detail state feeding a coordinate decoder.

Critical anti-cheat: text is NEVER supervised against pixels and never enters the
image decoder directly. Text-to-image completion must arise because text-only and
image-only encounters are trained to occupy compatible PUBLIC block states. The
image decoder learns to reconstruct images from image-derived public/private state;
with private detail removed, it learns the class-level visual family/prototype.

This is Gate 0X0: establish cross-modal completion first. Autonomous SEEK/LOCK is a
later gate, not smuggled into this experiment.
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
    DEFAULT_ALPHABET,
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
PROMPT_TEMPLATES = [
    "{}",
    "a {}",
    "a photo of a {}",
    "an image of a {}",
    "clothing item {}",
]


class FashionPairs(Dataset):
    def __init__(self, hf_dataset, *, augment_text: bool = True) -> None:
        self.ds = hf_dataset
        self.augment_text = augment_text

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> tuple[Tensor, int, str]:
        item = self.ds[index]
        image = item["image"].convert("L")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)
        label = int(item["label"])
        if self.augment_text:
            template = random.choice(PROMPT_TEMPLATES)
            text = template.format(LABELS[label])
        else:
            text = LABELS[label]
        return x, label, text


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


def class_prompts(labels: Tensor, *, shuffled: bool) -> list[str]:
    ids = labels.detach().cpu().tolist()
    if shuffled and len(ids) > 1:
        # Derangement-like roll: destroys text/image pairing without changing the
        # marginal frequency of either modality.
        ids = ids[1:] + ids[:1]
    return [random.choice(PROMPT_TEMPLATES).format(LABELS[i]) for i in ids]


def save_grid(images: Tensor, path: Path, *, row_labels: list[str] | None = None, scale: int = 4) -> None:
    """Save [rows, cols, 1, H, W] or [rows, 1, H, W] as a labelled PNG."""
    x = images.detach().float().cpu().clamp(0, 1)
    if x.ndim == 4:
        x = x[:, None]
    if x.ndim != 5:
        raise ValueError("images must be [rows, cols, 1, H, W] or [rows, 1, H, W]")
    rows, cols, _, h, w = x.shape
    label_w = 110 if row_labels else 0
    header_h = 18
    canvas = Image.new("L", (label_w + cols * w * scale, header_h + rows * h * scale), 255)
    draw = ImageDraw.Draw(canvas)
    for c in range(cols):
        draw.text((label_w + c * w * scale + 2, 2), str(c + 1), fill=0)
    for r in range(rows):
        if row_labels:
            draw.text((2, header_h + r * h * scale + 4), row_labels[r], fill=0)
        for c in range(cols):
            arr = (x[r, c, 0].numpy() * 255.0).astype(np.uint8)
            tile = Image.fromarray(arr, mode="L").resize((w * scale, h * scale), Image.Resampling.NEAREST)
            canvas.paste(tile, (label_w + c * w * scale, header_h + r * h * scale))
    canvas.save(path)


def decode_text_trajectory(
    model: CrossModalBlockModel,
    tokenizer: CharTokenizer,
    texts: list[str],
    *,
    device: torch.device,
    phase_mode: str = "dynamic",
) -> Tensor:
    tokens = tokenizer.encode(texts, device=device)
    encoded = model.encode(tokens=tokens, phase_mode=phase_mode, return_trace=True)
    trace = encoded["trace"]
    assert isinstance(trace, dict)
    states = trace["states"]  # [B, steps, state]
    frames = []
    for step in range(states.shape[1]):
        # Text has no instance evidence, so explicitly collapse private detail.
        frames.append(model.decode_state(states[:, step], keep_detail=False))
    return torch.stack(frames, dim=1)


def compute_losses(
    model: CrossModalBlockModel,
    tokenizer: CharTokenizer,
    image: Tensor,
    label: Tensor,
    texts: list[str],
    *,
    phase_mode: str,
) -> tuple[Tensor, dict[str, float]]:
    tokens = tokenizer.encode(texts, device=image.device)

    # SAME substrate, three operating conditions.
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
    assert all(isinstance(v, Tensor) for v in [pub_t, pub_i, pub_p, det_t, det_i, det_p, logits_i, logits_p])

    # Pixel learning comes ONLY from visual evidence or paired evidence.
    recon_i = model.decode(pub_i, det_i)
    recon_p = model.decode(pub_p, det_p)

    # The public-only decoder is trained on image-derived public state with private
    # detail erased. Across many examples of a class, MSE therefore asks it for the
    # stable class-level visual family/prototype. Text never gets this pixel loss.
    coarse_i = model.decode(pub_i, None)

    loss_recon = F.mse_loss(recon_i, image)
    loss_paired_recon = F.mse_loss(recon_p, image)
    loss_coarse = F.mse_loss(coarse_i, image)

    # CLIP-like relation, but after each modality has passed through the SAME block.
    loss_align = (1.0 - F.cosine_similarity(pub_t, pub_i.detach(), dim=-1)).mean()
    loss_pair_align = 0.5 * (
        (1.0 - F.cosine_similarity(pub_p, pub_i.detach(), dim=-1)).mean()
        + (1.0 - F.cosine_similarity(pub_p, pub_t.detach(), dim=-1)).mean()
    )

    # Prevent trivial public collapse using only the IMAGE label. Text receives no
    # class-index supervision; it must reach the image-trained public geometry.
    loss_cls = F.cross_entropy(logits_i, label)
    loss_pair_cls = F.cross_entropy(logits_p, label)

    # Text-only state should not invent instance-specific private detail.
    loss_text_detail = det_t.square().mean()
    loss_detail_size = 0.05 * det_i.square().mean()

    loss = (
        1.00 * loss_recon
        + 0.70 * loss_paired_recon
        + 0.45 * loss_coarse
        + 0.80 * loss_align
        + 0.25 * loss_pair_align
        + 0.35 * loss_cls
        + 0.15 * loss_pair_cls
        + 0.08 * loss_text_detail
        + loss_detail_size
    )

    metrics = {
        "loss": float(loss.detach()),
        "recon": float(loss_recon.detach()),
        "paired_recon": float(loss_paired_recon.detach()),
        "coarse": float(loss_coarse.detach()),
        "align": float(loss_align.detach()),
        "image_cls": float(loss_cls.detach()),
    }
    return loss, metrics


@torch.no_grad()
def evaluate(
    model: CrossModalBlockModel,
    tokenizer: CharTokenizer,
    loader: DataLoader,
    *,
    device: torch.device,
    phase_mode: str,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total = 0
    image_correct = 0
    text_correct = 0
    recon_sum = 0.0
    coarse_sum = 0.0
    align_sum = 0.0

    for batch_idx, (image, label, _) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        texts = [LABELS[i] for i in label.cpu().tolist()]
        tokens = tokenizer.encode(texts, device=device)

        text_only = model.encode(tokens=tokens, phase_mode=phase_mode)
        image_only = model.encode(image=image, phase_mode=phase_mode)
        pub_t = text_only["public"]
        pub_i = image_only["public"]
        det_i = image_only["detail"]
        logits_t = text_only["logits"]
        logits_i = image_only["logits"]
        assert all(isinstance(v, Tensor) for v in [pub_t, pub_i, det_i, logits_t, logits_i])

        recon = model.decode(pub_i, det_i)
        coarse = model.decode(pub_i, None)
        b = image.shape[0]
        total += b
        image_correct += int((logits_i.argmax(dim=-1) == label).sum())
        text_correct += int((logits_t.argmax(dim=-1) == label).sum())
        recon_sum += float(F.mse_loss(recon, image, reduction="sum"))
        coarse_sum += float(F.mse_loss(coarse, image, reduction="sum"))
        align_sum += float((1.0 - F.cosine_similarity(pub_t, pub_i, dim=-1)).sum())

    pixels = total * model.config.image_size * model.config.image_size
    return {
        "image_to_concept_acc": image_correct / max(total, 1),
        "text_to_concept_acc": text_correct / max(total, 1),
        "image_recon_mse": recon_sum / max(pixels, 1),
        "public_only_mse": coarse_sum / max(pixels, 1),
        "crossmodal_cosine_error": align_sum / max(total, 1),
    }


@torch.no_grad()
def text_prototype_report(
    model: CrossModalBlockModel,
    tokenizer: CharTokenizer,
    *,
    device: torch.device,
    phase_mode: str,
) -> tuple[Tensor, list[int]]:
    tokens = tokenizer.encode(LABELS, device=device)
    out = model.encode(tokens=tokens, phase_mode=phase_mode)
    pub = out["public"]
    logits = out["logits"]
    assert isinstance(pub, Tensor) and isinstance(logits, Tensor)
    images = model.decode(pub, None)
    predicted = logits.argmax(dim=-1).cpu().tolist()
    return images, predicted


def print_metrics(prefix: str, metrics: dict[str, float]) -> None:
    body = "  ".join(f"{k}={v:.5f}" for k, v in metrics.items())
    print(f"{prefix}: {body}")


def load_checkpoint(path: Path, device: torch.device) -> tuple[CrossModalBlockModel, CharTokenizer, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = CrossModalConfig(**payload["config"])
    alphabet = payload.get("alphabet", DEFAULT_ALPHABET)
    model = CrossModalBlockModel(config, alphabet=alphabet).to(device)
    model.load_state_dict(payload["model"])
    tokenizer = CharTokenizer(alphabet=alphabet, max_len=config.max_text_len)
    return model, tokenizer, payload


def run_query(args: argparse.Namespace, device: torch.device) -> None:
    if args.checkpoint is None:
        raise SystemExit("--query requires --checkpoint")
    model, tokenizer, _ = load_checkpoint(Path(args.checkpoint), device)
    model.eval()
    texts = [q.strip() for q in args.query.split("|") if q.strip()]
    with torch.no_grad():
        frames = decode_text_trajectory(model, tokenizer, texts, device=device)
        tokens = tokenizer.encode(texts, device=device)
        out = model.encode(tokens=tokens)
        logits = out["logits"]
        assert isinstance(logits, Tensor)
        probs = logits.softmax(dim=-1)
        pred = probs.argmax(dim=-1)
        for i, text in enumerate(texts):
            print(
                f"query={text!r} -> public concept={LABELS[int(pred[i])]} "
                f"confidence={float(probs[i, pred[i]]):.3f}"
            )
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_grid(frames, out_path / "query_trajectory.png", row_labels=texts, scale=args.scale)
    print(f"saved {out_path / 'query_trajectory.png'}")


def train_main(args: argparse.Namespace, device: torch.device) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Gate 0X training needs Hugging Face datasets. Install with: pip install -e '.[crossmodal]'"
        ) from exc

    print(f"device={device}")
    print(f"dataset={DATASET_NAME}")
    print("loading Hugging Face Fashion-MNIST...")
    train_hf = load_dataset(DATASET_NAME, split="train")
    test_hf = load_dataset(DATASET_NAME, split="test")
    train_hf = limit_dataset(train_hf, args.train_limit, args.seed)
    test_hf = limit_dataset(test_hf, args.test_limit, args.seed + 1)

    train_ds = FashionPairs(train_hf, augment_text=True)
    test_ds = FashionPairs(test_hf, augment_text=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_text_acc = -1.0

    print(f"train rows={len(train_ds)}  test rows={len(test_ds)}")
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}")
    print(
        "anti-cheat: no text->pixel loss, no text->decoder skip, no concept phase labels; "
        "text must align to image-trained public state"
    )
    if args.shuffle_pairs:
        print("CONTROL ACTIVE: text/image pairings are shuffled within every training batch")

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums: dict[str, float] = {}
        seen_batches = 0
        for image, label, _ in train_loader:
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            texts = class_prompts(label, shuffled=args.shuffle_pairs)
            loss, metrics = compute_losses(
                model,
                tokenizer,
                image,
                label,
                texts,
                phase_mode=args.phase_mode,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + v
            seen_batches += 1
            if args.max_batches and seen_batches >= args.max_batches:
                break

        train_metrics = {k: v / max(seen_batches, 1) for k, v in sums.items()}
        eval_dynamic = evaluate(
            model,
            tokenizer,
            test_loader,
            device=device,
            phase_mode=args.phase_mode,
            max_batches=args.eval_batches,
        )
        # Mechanism ablation: same learned weights, but remove traversal.
        eval_clamped = evaluate(
            model,
            tokenizer,
            test_loader,
            device=device,
            phase_mode="clamped",
            max_batches=args.eval_batches,
        )
        print_metrics(f"epoch {epoch:02d} train", train_metrics)
        print_metrics(f"epoch {epoch:02d} eval/{args.phase_mode}", eval_dynamic)
        print_metrics(f"epoch {epoch:02d} eval/clamped", eval_clamped)

        images, predicted = text_prototype_report(
            model, tokenizer, device=device, phase_mode=args.phase_mode
        )
        save_grid(images, output_dir / "text_prototypes.png", row_labels=LABELS, scale=args.scale)
        frames = decode_text_trajectory(model, tokenizer, LABELS, device=device, phase_mode=args.phase_mode)
        save_grid(frames, output_dir / "text_trajectory.png", row_labels=LABELS, scale=args.scale)

        checkpoint = {
            "model": model.state_dict(),
            "config": asdict(config),
            "alphabet": tokenizer.alphabet,
            "labels": LABELS,
            "dataset": DATASET_NAME,
            "epoch": epoch,
            "args": vars(args),
            "eval": eval_dynamic,
            "eval_clamped": eval_clamped,
            "text_query_predictions": predicted,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if eval_dynamic["text_to_concept_acc"] > best_text_acc:
            best_text_acc = eval_dynamic["text_to_concept_acc"]
            torch.save(checkpoint, output_dir / "best.pt")

    print("\nGate 0X0 training complete")
    print(f"best text->image-public concept accuracy={best_text_acc:.4f}")
    print(f"checkpoint: {output_dir / 'best.pt'}")
    print(f"prototype sheet: {output_dir / 'text_prototypes.png'}")
    print(f"trajectory sheet: {output_dir / 'text_trajectory.png'}")
    print("Next controls: --phase-mode clamped, then --shuffle-pairs. Do not interpret one successful run alone.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--train-limit", type=int, default=12_000)
    parser.add_argument("--test-limit", type=int, default=2_000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=18_001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="runs/gate0x_fashion")
    parser.add_argument("--phase-mode", choices=["dynamic", "clamped"], default="dynamic")
    parser.add_argument("--shuffle-pairs", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0, help="debug: cap training batches per epoch")
    parser.add_argument("--eval-batches", type=int, default=None, help="optional evaluation batch cap")

    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--phase-dim", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--public-dim", type=int, default=24)
    parser.add_argument("--detail-dim", type=int, default=32)
    parser.add_argument("--basis-dim", type=int, default=128)

    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--query",
        default=None,
        help="query a checkpoint with raw text; separate multiple queries with |",
    )
    parser.add_argument("--scale", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    if args.query:
        run_query(args, device)
    else:
        train_main(args, device)


if __name__ == "__main__":
    main()
