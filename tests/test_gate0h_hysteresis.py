from __future__ import annotations

import torch

from blockneuron.hysteresis import HysteresisConfig, LandauBranchMatter, SchmittLatch
from experiments.gate0h_hysteretic_matter import PROGRAM_A, PROGRAM_B, balanced_programs


def matter() -> LandauBranchMatter:
    return LandauBranchMatter(HysteresisConfig(dt=0.08, substeps=15, settle_steps=100))


def test_same_pulses_different_order_write_different_stable_states() -> None:
    m = matter()
    assert sorted(PROGRAM_A.tolist()) == sorted(PROGRAM_B.tolist())
    a, _ = m.run_program(torch.tensor(-1.0), PROGRAM_A)
    b, _ = m.run_program(torch.tensor(-1.0), PROGRAM_B)
    a = m.relax(a)
    b = m.relax(b)
    assert float(a) < -0.99
    assert float(b) > 0.99
    assert float(m.conductance(a)) < 0.05
    assert float(m.conductance(b)) > 0.95


def test_hysteretic_state_holds_without_silent_updates_and_can_be_erased() -> None:
    m = matter()
    b, _ = m.run_program(torch.tensor(-1.0), PROGRAM_B)
    b = m.relax(b)
    held, updates = m.retain(b, 1_000_000)
    assert updates == 0
    assert torch.equal(held, b)

    erased = m.integrate(held, -1.4)
    erased = m.relax(erased)
    assert float(erased) < -0.99


def test_subthreshold_returns_to_same_well_and_strong_pulse_flips() -> None:
    m = matter()
    p0 = torch.tensor(-1.0)
    sub = m.relax(m.integrate(p0, 0.20))
    supra = m.relax(m.integrate(p0, 1.20))
    assert float(sub) < -0.99
    assert float(supra) > 0.99


def test_landau_program_is_differentiable_before_discrete_readout() -> None:
    m = matter()
    fields = PROGRAM_B.clone().requires_grad_(True)
    p, _ = m.run_program(torch.tensor(-1.0), fields)
    p.backward()
    assert fields.grad is not None
    assert float(fields.grad.norm()) > 1e-5


def test_explicit_latch_is_mandatory_tying_control() -> None:
    latch = SchmittLatch(1.0)
    a, _ = latch.run_program(torch.tensor(-1.0), PROGRAM_A)
    b, _ = latch.run_program(torch.tensor(-1.0), PROGRAM_B)
    assert float(a) == -1.0
    assert float(b) == 1.0
    held, updates = latch.retain(b, 100_000)
    assert float(held) == 1.0
    assert updates == 0


def test_noisy_program_batch_shapes_and_balance() -> None:
    fields, labels = balanced_programs(100, sigma=0.05, seed=3)
    assert fields.shape == (100, 3)
    assert labels.shape == (100,)
    assert int(labels.sum()) == 50
