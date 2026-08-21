from __future__ import annotations

"""Gate 0H1 — computation leaves a reusable configuration behind.

A brief programming event writes one of four operations into hysteretic local
branch state. The programming signal is then removed. Many later data inputs
must be processed by the already-configured substrate with zero material-state
updates between reads.

Controls:
  * explicit latch: same write-once capability, proving memory is not unique;
  * context mux: exact operation context must be supplied on every read;
  * clocked state keeper: exact state, but one explicit state transition/read.

H1 therefore tests temporal organization and amortization, not unique
expressivity or physical energy.
"""

import argparse
import math
import sys
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blockneuron.configured_substrate import (  # noqa: E402
    ClockedOperationKeeper,
    ContextOperationMux,
    ExplicitOperationLatch,
    HystereticOperationSubstrate,
    OPERATION_NAMES,
    apply_matrix_bank,
    operation_matrices,
)


def nearest_operation_accuracy(x: Tensor, y: Tensor, target_op: int) -> float:
    mats = operation_matrices(dtype=x.dtype, device=x.device)
    candidates = torch.stack(
        [apply_matrix_bank(x, mats, op) for op in range(len(OPERATION_NAMES))],
        dim=1,
    )
    mse = (candidates - y[:, None, :]).square().mean(dim=-1)
    pred = mse.argmin(dim=1)
    return float((pred == int(target_op)).float().mean())


def deterministic_receipt(samples: int, seed: int) -> dict[str, float]:
    g = torch.Generator().manual_seed(seed)
    substrate = HystereticOperationSubstrate()
    latch = ExplicitOperationLatch()
    context = ContextOperationMux()
    clocked = ClockedOperationKeeper()
    mats = operation_matrices()

    state = substrate.initial_state()
    x = torch.randn(samples, 4, generator=g)

    max_hard_mse = 0.0
    max_soft_mse = 0.0
    max_latch_mse = 0.0
    max_context_mse = 0.0
    max_clocked_mse = 0.0
    min_soft_op_acc = 1.0
    switch_work = []
    write_updates = []

    print("Gate 0H1 deterministic receipt")
    print("configure once -> remove program -> reuse configured operation")
    print()
    print("operation   selected  retained  hard_mse     soft_mse     switch_proxy")

    for op, name in enumerate(OPERATION_NAMES):
        state, receipt = substrate.program(state, op)
        retained, silent_updates = substrate.retain(state, 10_000)
        target = apply_matrix_bank(x, mats, op)

        y_hard = substrate.apply(x, retained)
        y_soft = substrate.apply_soft(x, retained)

        latch_state = latch.program(op)
        latch_state, latch_updates = latch.retain(latch_state, 10_000)
        y_latch = latch.apply(x, latch_state)
        y_context = context.apply(x, op)

        recurrent_state = clocked.program(op)
        y_clocked, recurrent_state, clocked_updates = clocked.read(x, recurrent_state)

        hard_mse = float((y_hard - target).square().mean())
        soft_mse = float((y_soft - target).square().mean())
        latch_mse = float((y_latch - target).square().mean())
        context_mse = float((y_context - target).square().mean())
        clocked_mse = float((y_clocked - target).square().mean())
        soft_acc = nearest_operation_accuracy(x, y_soft, op)

        max_hard_mse = max(max_hard_mse, hard_mse)
        max_soft_mse = max(max_soft_mse, soft_mse)
        max_latch_mse = max(max_latch_mse, latch_mse)
        max_context_mse = max(max_context_mse, context_mse)
        max_clocked_mse = max(max_clocked_mse, clocked_mse)
        min_soft_op_acc = min(min_soft_op_acc, soft_acc)
        switch_work.append(receipt.switch_work_proxy)
        write_updates.append(receipt.internal_scalar_updates)

        retained_ok = substrate.selected_operation(retained) == op
        assert silent_updates == 0
        assert latch_updates == 0
        assert clocked_updates == 1
        print(
            f"{name:9s} {receipt.selected_operation:^9d}  "
            f"{str(retained_ok):^8s}  {hard_mse:10.3e}  {soft_mse:10.3e}  "
            f"{receipt.switch_work_proxy:10.4f}"
        )

    print()
    print("retention horizon=10,000 silent ticks; executed hysteretic state updates=0")
    print(f"max hard-route MSE       : {max_hard_mse:.3e}")
    print(f"max soft-route MSE       : {max_soft_mse:.3e}")
    print(f"min soft-route op acc    : {min_soft_op_acc:.4f}")
    print(f"explicit latch max MSE   : {max_latch_mse:.3e}")
    print(f"context mux max MSE      : {max_context_mse:.3e}")
    print(f"clocked keeper max MSE   : {max_clocked_mse:.3e}")
    print(f"mean switch-work proxy   : {sum(switch_work)/len(switch_work):.4f}")
    print(f"scalar integration updates/write (software simulation): {write_updates[0]}")

    # Reconfiguration receipt: state is currently FILTER. Rewrite directly to NEGATE.
    state, reprogram = substrate.program(state, 1)
    after, silent_updates = substrate.retain(state, 1_000_000)
    reprogram_ok = substrate.selected_operation(after) == 1
    print()
    print(
        "reprogram FILTER -> NEGATE -> silence 1,000,000: "
        f"selected={substrate.selected_operation(after)} ok={reprogram_ok} "
        f"silent_updates={silent_updates}"
    )

    return {
        "max_hard_mse": max_hard_mse,
        "max_soft_mse": max_soft_mse,
        "min_soft_op_acc": min_soft_op_acc,
        "reprogram_ok": float(reprogram_ok),
        "write_scalar_updates": float(write_updates[0]),
    }


def noise_receipt(trials: int, seed: int) -> dict[float, float]:
    substrate = HystereticOperationSubstrate()
    g = torch.Generator().manual_seed(seed + 991)
    sigmas = [0.00, 0.05, 0.10, 0.20, 0.30]
    result: dict[float, float] = {}

    print()
    print("noisy reprogramming accuracy")
    print("sigma    hysteretic    explicit-latch")
    for sigma in sigmas:
        correct = 0
        for _ in range(trials):
            previous = int(torch.randint(0, len(OPERATION_NAMES), (1,), generator=g).item())
            target = int(torch.randint(0, len(OPERATION_NAMES), (1,), generator=g).item())
            state = substrate.initial_state()
            state, _ = substrate.program(state, previous)
            state, _ = substrate.program(state, target, noise_std=sigma, generator=g)
            state, updates = substrate.retain(state, 10_000)
            assert updates == 0
            correct += int(substrate.selected_operation(state) == target)
        acc = correct / trials
        result[sigma] = acc
        print(f"{sigma:4.2f}      {acc:8.4f}         1.0000")
    return result


def amortization_receipt(write_scalar_updates: float) -> None:
    """Report accounting proxies; none of these are physical energy measurements."""
    control_width = float(len(OPERATION_NAMES))
    print()
    print("control / maintenance accounting (proxies, NOT joules)")
    print(
        "uses      H external control/use   H simulated write-updates/use   "
        "latch control/use   context control/use   clocked state updates/use"
    )
    for uses in [1, 10, 100, 1_000, 10_000, 1_000_000]:
        h_control = control_width / uses
        h_write = write_scalar_updates / uses
        latch_control = control_width / uses
        context_control = control_width
        clocked_updates = 1.0
        print(
            f"{uses:7d}          {h_control:10.6f}                 {h_write:10.6f}"
            f"             {latch_control:10.6f}             {context_control:10.6f}"
            f"                 {clocked_updates:8.4f}"
        )

    print()
    print(
        "Interpretation: H and the explicit latch pay a one-time configuration "
        "cost and require no mandatory state transition between reads. The "
        "context mux pays control bandwidth on every read. The clocked keeper "
        "executes a state transition every read by construction; an event-driven "
        "keeper can skip that transition and then becomes latch-like."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=19_001)
    args = parser.parse_args()

    receipt = deterministic_receipt(args.samples, args.seed)
    noise = noise_receipt(args.trials, args.seed)
    amortization_receipt(receipt["write_scalar_updates"])

    passed = (
        receipt["max_hard_mse"] < 1e-12
        and receipt["max_soft_mse"] < 1e-4
        and receipt["min_soft_op_acc"] > 0.999
        and receipt["reprogram_ok"] == 1.0
        and noise[0.10] >= 0.95
    )
    print()
    print(f"Gate 0H1 mechanism verdict: {'PASS' if passed else 'FAIL'}")
    print(
        "Claim boundary: persistent write-once computational configuration and "
        "amortized control are established if this passes. The explicit latch "
        "ties the capability, so unique expressivity and physical-energy claims "
        "are NOT established."
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
