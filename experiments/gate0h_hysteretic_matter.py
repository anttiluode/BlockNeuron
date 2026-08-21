from __future__ import annotations

"""Gate 0H — hysteretic branch matter: write, hold, probe, erase.

Two pulse programs contain exactly the same three fields but in a different
order. A continuous double-well branch state can end in different persistent
configurations because crossing the barrier is history dependent. The current
field is then removed completely before readout.

This is a mechanism receipt, not a novelty claim. A plain explicit Schmitt latch
is included and should tie the persistent-memory capability. The interesting
object is the differentiable local branch configuration, not the statement that
latches can remember bits.
"""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blockneuron.hysteresis import (  # noqa: E402
    HysteresisConfig,
    LandauBranchMatter,
    MemorylessGate,
    SchmittLatch,
)

PROGRAM_A = torch.tensor([+1.20, -1.20, +0.80])
PROGRAM_B = torch.tensor([-1.20, +1.20, +0.80])


def balanced_programs(trials: int, *, sigma: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if trials % 2:
        raise ValueError("trials must be even")
    g = torch.Generator().manual_seed(seed)
    half = trials // 2
    programs = torch.stack(
        [PROGRAM_A.repeat(half, 1), PROGRAM_B.repeat(half, 1)], dim=0
    ).reshape(trials, -1)
    labels = torch.cat(
        [torch.zeros(half, dtype=torch.long), torch.ones(half, dtype=torch.long)]
    )
    if sigma > 0:
        programs = programs + sigma * torch.randn(programs.shape, generator=g)
    order = torch.randperm(trials, generator=g)
    return programs[order], labels[order]


def accuracy_from_sign(state: torch.Tensor, labels: torch.Tensor) -> float:
    pred = (state > 0).long()
    return float((pred == labels).float().mean())


def run_noise_sweep(args: argparse.Namespace) -> list[dict[str, float]]:
    matter = LandauBranchMatter(
        HysteresisConfig(
            dt=args.dt,
            substeps=args.substeps,
            settle_steps=args.settle_steps,
            conductance_gain=args.conductance_gain,
        )
    )
    latch = SchmittLatch(args.latch_threshold)
    memoryless = MemorylessGate(args.conductance_gain)
    rows: list[dict[str, float]] = []

    for index, sigma in enumerate(args.noise):
        fields, labels = balanced_programs(
            args.trials, sigma=sigma, seed=args.seed + index
        )
        p0 = torch.tensor(-1.0)
        p_written, _ = matter.run_program(p0, fields)
        p_settled = matter.relax(p_written)
        landau_acc = accuracy_from_sign(p_settled, labels)

        latch_state, _ = latch.run_program(torch.tensor(-1.0), fields)
        latch_acc = accuracy_from_sign(latch_state, labels)

        # After the program the external field is exactly zero for both classes.
        # A current-context gate therefore has no information about traversal.
        current = memoryless.read(torch.zeros(args.trials))
        memoryless_pred = (current > 0.5).long()
        memoryless_acc = float((memoryless_pred == labels).float().mean())

        rows.append(
            {
                "sigma": float(sigma),
                "landau_acc": landau_acc,
                "latch_acc": latch_acc,
                "memoryless_acc": memoryless_acc,
            }
        )
    return rows


def deterministic_receipt(args: argparse.Namespace) -> dict[str, float]:
    matter = LandauBranchMatter(
        HysteresisConfig(
            dt=args.dt,
            substeps=args.substeps,
            settle_steps=args.settle_steps,
            conductance_gain=args.conductance_gain,
        )
    )

    p0 = torch.tensor(-1.0)
    p_a_write, trace_a = matter.run_program(p0, PROGRAM_A)
    p_b_write, trace_b = matter.run_program(p0, PROGRAM_B)
    p_a = matter.relax(p_a_write)
    p_b = matter.relax(p_b_write)
    g_a = matter.conductance(p_a)
    g_b = matter.conductance(p_b)

    # Retention is identity after the state has fallen into a stable well.
    retained = {}
    for silent_steps in args.silence:
        a_hold, a_updates = matter.retain(p_a, silent_steps)
        b_hold, b_updates = matter.retain(p_b, silent_steps)
        retained[silent_steps] = (
            float(a_hold),
            float(b_hold),
            a_updates + b_updates,
        )

    # Erase/rewrite the B-written positive state with one strong opposite pulse.
    erased = matter.integrate(p_b, args.erase_field)
    erased = matter.relax(erased)

    # Subthreshold perturbation should leave the basin unchanged; strong drive flips.
    sub = matter.integrate(torch.tensor(-1.0), args.subthreshold_field)
    sub = matter.relax(sub)
    supra = matter.integrate(torch.tensor(-1.0), args.suprathreshold_field)
    supra = matter.relax(supra)

    # Differentiability receipt: final pre-settle state depends on pulse amplitudes.
    differentiable_fields = PROGRAM_B.clone().requires_grad_(True)
    p_grad, _ = matter.run_program(torch.tensor(-1.0), differentiable_fields)
    p_grad.backward()
    grad_norm = float(differentiable_fields.grad.norm())

    print("Gate 0H deterministic receipt")
    print(f"program A={PROGRAM_A.tolist()}")
    print(f"program B={PROGRAM_B.tolist()}")
    print("same pulse multiset:", sorted(PROGRAM_A.tolist()) == sorted(PROGRAM_B.tolist()))
    print(f"A after write={float(p_a_write):+.4f}  after settle={float(p_a):+.4f}  conductance={float(g_a):.4f}")
    print(f"B after write={float(p_b_write):+.4f}  after settle={float(p_b):+.4f}  conductance={float(g_b):.4f}")
    print(f"order gap after settle={float((p_b - p_a).abs()):.4f}")
    print("retention after drive removed:")
    for silent_steps, (a_hold, b_hold, updates) in retained.items():
        print(
            f"  silence={silent_steps:>7d}: A={a_hold:+.4f} B={b_hold:+.4f} "
            f"executed_silent_state_updates={updates}"
        )
    print(f"erase field={args.erase_field:+.2f}: B state -> {float(erased):+.4f}")
    print(
        f"threshold receipt: sub={args.subthreshold_field:+.2f} -> {float(sub):+.4f}; "
        f"supra={args.suprathreshold_field:+.2f} -> {float(supra):+.4f}"
    )
    print(f"pulse-gradient norm={grad_norm:.6f}")
    print("trace A:", " ".join(f"{float(v):+.3f}" for v in trace_a))
    print("trace B:", " ".join(f"{float(v):+.3f}" for v in trace_b))

    return {
        "p_a": float(p_a),
        "p_b": float(p_b),
        "g_a": float(g_a),
        "g_b": float(g_b),
        "erase": float(erased),
        "subthreshold": float(sub),
        "suprathreshold": float(supra),
        "gradient_norm": grad_norm,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--substeps", type=int, default=15)
    parser.add_argument("--settle-steps", type=int, default=100)
    parser.add_argument("--conductance-gain", type=float, default=4.0)
    parser.add_argument("--latch-threshold", type=float, default=1.0)
    parser.add_argument("--erase-field", type=float, default=-1.4)
    parser.add_argument("--subthreshold-field", type=float, default=0.20)
    parser.add_argument("--suprathreshold-field", type=float, default=1.20)
    parser.add_argument("--silence", type=int, nargs="+", default=[0, 10, 100, 10_000])
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--trials", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=18_021)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = deterministic_receipt(args)
    rows = run_noise_sweep(args)

    print("\nnoise robustness (program classification after silence)")
    print("sigma    Landau    explicit-latch    memoryless")
    for row in rows:
        print(
            f"{row['sigma']:>5.2f}    {row['landau_acc']:.4f}       "
            f"{row['latch_acc']:.4f}          {row['memoryless_acc']:.4f}"
        )

    passed = (
        receipt["p_a"] < -0.9
        and receipt["p_b"] > 0.9
        and receipt["g_a"] < 0.05
        and receipt["g_b"] > 0.95
        and receipt["erase"] < -0.9
        and receipt["subthreshold"] < -0.9
        and receipt["suprathreshold"] > 0.9
        and receipt["gradient_norm"] > 1e-5
        and rows[0]["landau_acc"] > 0.99
        and rows[0]["latch_acc"] > 0.99
        and abs(rows[0]["memoryless_acc"] - 0.5) < 1e-6
    )
    print("\nGate 0H verdict:", "PASS" if passed else "FAIL")
    print(
        "Interpretation: history-dependent local configuration, retention and erasure "
        "are established. The explicit latch ties the memory capability, so unique "
        "computational expressivity is NOT established."
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
