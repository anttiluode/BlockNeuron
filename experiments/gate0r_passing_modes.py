from __future__ import annotations

"""Gate 0R — rhythmic phase as an ordered program, not a clean address.

The same four overlapping phase-conditioned operators are traversed in two orders.
Both programs begin and end at phase zero and use the same multiset of phase
anchors. Their outputs differ only because the internal state experiences the
operators in a different order.

This is a mechanism receipt, not a uniqueness claim. A matched 16-parameter
Fourier-conditioned recurrent attacker solves the toy too.
"""

import argparse
import math
import random

import numpy as np
import torch
from torch import Tensor, nn


ANCHORS = torch.tensor([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
FORWARD = [0, 1, 2, 3, 0]
REVERSE = [0, 3, 2, 1, 0]

# Four deliberately non-commuting primitive transforms. The learner does not
# need to recover these matrices individually; it only has to recover the two
# whole-cycle programs they define.
TEACHER = torch.tensor(
    [
        [[1.00, 0.70], [0.00, 1.00]],
        [[1.00, 0.00], [0.60, 1.00]],
        [[0.90, -0.40], [0.40, 0.90]],
        [[1.15, 0.00], [0.00, 0.85]],
    ],
    dtype=torch.float32,
)


def compose(mats: Tensor, order: list[int]) -> Tensor:
    """Return the ordered product acting on a column vector."""
    out = torch.eye(mats.shape[-1], dtype=mats.dtype, device=mats.device)
    for k in order:
        out = mats[k] @ out
    return out


class PassingModeCycle(nn.Module):
    """Four overlapping phase windows expose passing effective operators."""

    def __init__(self, kappa: float = 3.0) -> None:
        super().__init__()
        self.kappa = float(kappa)
        self.branch = nn.Parameter(
            torch.eye(2).repeat(4, 1, 1) + 0.05 * torch.randn(4, 2, 2)
        )
        self.register_buffer("anchors", ANCHORS.clone())

    def gates(self, phase: Tensor) -> Tensor:
        # Von-Mises-like phase windows. kappa=3 is intentionally soft: at an
        # anchor the dominant mode is ~0.907 and each neighbor is ~0.045.
        return torch.softmax(
            self.kappa * torch.cos(phase[..., None] - self.anchors), dim=-1
        )

    def effective_matrix(self, phase: Tensor) -> Tensor:
        return torch.einsum("...b,bij->...ij", self.gates(phase), self.branch)

    def rollout(self, x: Tensor, order: list[int]) -> Tensor:
        h = x
        for k in order:
            h = h @ self.effective_matrix(self.anchors[k]).T
        return h


class FourierCycleAttacker(nn.Module):
    """Ordinary recurrent control: phase directly generates the step matrix.

    It has the same 16 trainable parameters as PassingModeCycle. At the four
    phase anchors the four Fourier features form a full-rank basis, so this is a
    deliberately strong attacker rather than a crippled baseline.
    """

    def __init__(self) -> None:
        super().__init__()
        self.coeff = nn.Parameter(torch.zeros(4, 2, 2))
        with torch.no_grad():
            self.coeff[0].copy_(torch.eye(2))
            self.coeff[1:].normal_(0.0, 0.05)

    def effective_matrix(self, phase: Tensor) -> Tensor:
        basis = torch.stack(
            [
                torch.ones_like(phase),
                torch.cos(phase),
                torch.sin(phase),
                torch.cos(2 * phase),
            ],
            dim=-1,
        )
        return torch.einsum("...b,bij->...ij", basis, self.coeff)

    def rollout(self, x: Tensor, order: list[int]) -> Tensor:
        h = x
        for k in order:
            h = h @ self.effective_matrix(ANCHORS[k]).T
        return h


def train(
    model: nn.Module,
    target_forward: Tensor,
    target_reverse: Tensor,
    *,
    steps: int,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=0.03)

    for _ in range(steps):
        x = torch.randn(512, 2)
        y_forward = x @ target_forward.T
        y_reverse = x @ target_reverse.T
        pred_forward = model.rollout(x, FORWARD)
        pred_reverse = model.rollout(x, REVERSE)
        loss = (pred_forward - y_forward).square().mean() + (
            pred_reverse - y_reverse
        ).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def evaluate(
    model: nn.Module, target_forward: Tensor, target_reverse: Tensor, *, seed: int
) -> tuple[float, float, float]:
    torch.manual_seed(seed + 10_000)
    x = torch.randn(10_000, 2)
    y_forward = x @ target_forward.T
    y_reverse = x @ target_reverse.T

    pred_forward = model.rollout(x, FORWARD)
    pred_reverse = model.rollout(x, REVERSE)
    fit = 0.5 * (
        (pred_forward - y_forward).square().mean()
        + (pred_reverse - y_reverse).square().mean()
    ).item()

    # Present the same two phase paths to the learned machine, but score each
    # against the target belonging to the opposite order.
    wrong_order = 0.5 * (
        (model.rollout(x, REVERSE) - y_forward).square().mean()
        + (model.rollout(x, FORWARD) - y_reverse).square().mean()
    ).item()

    # Destroy the passing modes: repeat only the phase-zero operator five times.
    h = x
    w0 = model.effective_matrix(torch.tensor(0.0))
    for _ in FORWARD:
        h = h @ w0.T
    phase_clamp = 0.5 * (
        (h - y_forward).square().mean() + (h - y_reverse).square().mean()
    ).item()
    return fit, wrong_order, phase_clamp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_200)
    parser.add_argument("--seed", type=int, default=18_001)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    target_forward = compose(TEACHER, FORWARD)
    target_reverse = compose(TEACHER, REVERSE)

    block = train(
        PassingModeCycle(kappa=3.0),
        target_forward,
        target_reverse,
        steps=args.steps,
        seed=args.seed,
    )
    attacker = train(
        FourierCycleAttacker(),
        target_forward,
        target_reverse,
        steps=args.steps,
        seed=args.seed + 1,
    )

    block_fit, block_wrong, block_clamp = evaluate(
        block, target_forward, target_reverse, seed=args.seed
    )
    attacker_fit, attacker_wrong, attacker_clamp = evaluate(
        attacker, target_forward, target_reverse, seed=args.seed + 1
    )

    with torch.no_grad():
        anchor_gates = torch.stack([block.gates(p) for p in ANCHORS])
        mean_max_gate = anchor_gates.max(dim=1).values.mean().item()
        adjacent_bleed = anchor_gates[0, 1].item()
        target_gap = torch.linalg.norm(target_forward - target_reverse).item()

    # Both programs end at phase zero. A memoryless endpoint-only system sees
    # the same x distribution and same endpoint phase for both, so its least-
    # squares optimum is simply the average of the two target matrices.
    endpoint_matrix = (target_forward + target_reverse) / 2
    torch.manual_seed(args.seed + 20_000)
    x = torch.randn(10_000, 2)
    y_forward = x @ target_forward.T
    y_reverse = x @ target_reverse.T
    endpoint_pred = x @ endpoint_matrix.T
    endpoint_mse = 0.5 * (
        (endpoint_pred - y_forward).square().mean()
        + (endpoint_pred - y_reverse).square().mean()
    ).item()

    print("Gate 0R — passing computational modes")
    print(f"teacher forward/reverse matrix gap : {target_gap:.6f}")
    print(f"phase-window mean max gate         : {mean_max_gate:.6f}")
    print(f"adjacent bleed at phase 0          : {adjacent_bleed:.6f}")
    print()
    print("model                 params       fit_mse      wrong_order    phase_clamp")
    print(
        f"PASSING-MODE BLOCK    {sum(p.numel() for p in block.parameters()):6d}   "
        f"{block_fit:10.6g}   {block_wrong:10.6g}   {block_clamp:10.6g}"
    )
    print(
        f"FOURIER RNN ATTACKER  {sum(p.numel() for p in attacker.parameters()):6d}   "
        f"{attacker_fit:10.6g}   {attacker_wrong:10.6g}   {attacker_clamp:10.6g}"
    )
    print(f"ENDPOINT-ONLY         {4:6d}   {endpoint_mse:10.6g}            -            -")
    print()

    mechanism_pass = (
        block_fit < 1e-4
        and block_wrong > 1e-2
        and block_clamp > 1e-2
        and adjacent_bleed > 1e-2
    )
    attacker_pass = attacker_fit < 1e-4
    print("ordered rhythmic mechanism:", "PASS" if mechanism_pass else "FAIL")
    print(
        "ordinary recurrent attacker also solves it:",
        "YES" if attacker_pass else "NO",
    )
    if not mechanism_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
