"""Gate 0: same structural mass, different effective machine."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from blockneuron import FourModeBlock, HyperLinearAttacker, UnconditionedLinear


def hadamard(n: int) -> Tensor:
    if n < 1 or n & (n - 1):
        raise ValueError("n must be a positive power of two")
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def rules(input_dim: int = 8) -> Tensor:
    return hadamard(input_dim)[:4] / math.sqrt(input_dim)


def sample(batch: int, *, input_dim: int = 8, device: str = "cpu") -> tuple[Tensor, ...]:
    x = torch.randn(batch, input_dim, device=device)
    chem_index = torch.randint(0, 2, (batch,), device=device)
    phase_index = torch.randint(0, 2, (batch,), device=device)
    chemical = chem_index.float() * 2.0 - 1.0
    phase = phase_index.float() * math.pi
    rule_index = chem_index * 2 + phase_index
    w = rules(input_dim).to(device)[rule_index]
    y = ((x * w).sum(-1) > 0).float()
    return x, chemical, phase, y, rule_index


@torch.no_grad()
def accuracy(model: nn.Module, batches: int = 10, batch: int = 4096, *, ablate: str = "none") -> float:
    correct = 0
    total = 0
    for _ in range(batches):
        x, chem, phase, y, _ = sample(batch)
        if ablate in {"chem", "both"}:
            chem = torch.zeros_like(chem)
        if ablate in {"phase", "both"}:
            phase = torch.full_like(phase, math.pi / 2.0)
        pred = model(x, chem, phase) > 0
        correct += (pred == y.bool()).sum().item()
        total += y.numel()
    return correct / total


def train(model: nn.Module, steps: int, seed: int, lr: float = 0.03) -> None:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        x, chem, phase, y, _ = sample(1024)
        loss = loss_fn(model(x, chem, phase), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


@dataclass
class Receipt:
    name: str
    accuracy: float
    no_chem: float
    no_phase: float
    no_context: float
    params: int


def run(seed: int = 18001, steps: int = 700) -> list[Receipt]:
    models = [
        ("BLOCK", FourModeBlock(8)),
        ("HYPER", HyperLinearAttacker(8)),
        ("PLAIN", UnconditionedLinear(8)),
    ]
    out: list[Receipt] = []
    for offset, (name, model) in enumerate(models):
        train(model, steps=steps, seed=seed + offset)
        out.append(
            Receipt(
                name=name,
                accuracy=accuracy(model),
                no_chem=accuracy(model, ablate="chem"),
                no_phase=accuracy(model, ablate="phase"),
                no_context=accuracy(model, ablate="both"),
                params=sum(p.numel() for p in model.parameters()),
            )
        )
    return out


def graph_separation(model: FourModeBlock) -> Tensor:
    chem = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    phase = torch.tensor([0.0, math.pi, 0.0, math.pi])
    w = model.effective_weight(chem, phase)
    w = torch.nn.functional.normalize(w, dim=-1)
    return w @ w.T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=18001)
    ap.add_argument("--steps", type=int, default=700)
    args = ap.parse_args()

    receipts = run(seed=args.seed, steps=args.steps)
    print("Gate 0 — SAME MASS / DIFFERENT MACHINE")
    print("name   params   full     -chem    -phase   -both")
    for r in receipts:
        print(
            f"{r.name:5s}  {r.params:6d}  {r.accuracy:7.4f}  {r.no_chem:7.4f}  "
            f"{r.no_phase:7.4f}  {r.no_context:7.4f}"
        )

    torch.manual_seed(args.seed)
    block = FourModeBlock(8)
    train(block, args.steps, args.seed)
    print("\nBLOCK effective-graph cosine matrix (four contexts):")
    print(graph_separation(block).detach().cpu().numpy().round(3))

    block_r = receipts[0]
    hyper_r = receipts[1]
    passes = block_r.accuracy > 0.97 and block_r.no_chem < 0.82 and block_r.no_phase < 0.82
    honest = hyper_r.accuracy > 0.97
    print(f"\nmechanism gate: {'PASS' if passes else 'FAIL'}")
    print(f"ordinary attacker also solves task: {'YES' if honest else 'NO'}")
    if passes and honest:
        print("verdict: structural instrument established; unique capability NOT established")


if __name__ == "__main__":
    main()
