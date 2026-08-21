from __future__ import annotations

import torch

from blockneuron.configured_substrate import (
    ContextOperationMux,
    ExplicitOperationLatch,
    HystereticOperationSubstrate,
    OPERATION_NAMES,
    apply_matrix_bank,
    operation_matrices,
)


def test_operation_bank_is_distinct() -> None:
    mats = operation_matrices()
    assert mats.shape == (4, 4, 4)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.allclose(mats[i], mats[j])


def test_hysteretic_substrate_programs_retains_and_reprograms() -> None:
    substrate = HystereticOperationSubstrate()
    state = substrate.initial_state()
    x = torch.randn(64, 4, generator=torch.Generator().manual_seed(7))
    mats = operation_matrices()

    for op in range(len(OPERATION_NAMES)):
        state, receipt = substrate.program(state, op)
        retained, updates = substrate.retain(state, 1_000_000)
        assert updates == 0
        assert receipt.selected_operation == op
        assert substrate.selected_operation(retained) == op

        target = apply_matrix_bank(x, mats, op)
        hard = substrate.apply(x, retained)
        soft = substrate.apply_soft(x, retained)
        assert torch.equal(hard, target)
        assert torch.mean((soft - target).square()).item() < 1e-4

    # Directly rewrite the final FILTER configuration to NEGATE.
    state, receipt = substrate.program(state, 1)
    assert receipt.selected_operation == 1
    assert substrate.selected_operation(state) == 1


def test_explicit_latch_and_context_mux_tie_capability() -> None:
    latch = ExplicitOperationLatch()
    mux = ContextOperationMux()
    mats = operation_matrices()
    x = torch.randn(32, 4, generator=torch.Generator().manual_seed(11))

    for op in range(len(OPERATION_NAMES)):
        target = apply_matrix_bank(x, mats, op)
        state = latch.program(op)
        retained, updates = latch.retain(state, 100_000)
        assert updates == 0
        assert torch.equal(latch.apply(x, retained), target)
        assert torch.equal(mux.apply(x, op), target)


def test_hysteretic_program_has_one_time_control_cost() -> None:
    substrate = HystereticOperationSubstrate()
    state, receipt = substrate.program(substrate.initial_state(), 2)
    assert receipt.external_control_values == len(OPERATION_NAMES)
    assert receipt.internal_scalar_updates > 0

    retained, updates = substrate.retain(state, 10_000)
    assert updates == 0
    assert substrate.selected_operation(retained) == 2

    # External control amortizes below the per-read four-value context by 2 uses.
    uses = 10
    hysteretic_control_per_use = receipt.external_control_values / uses
    context_control_per_use = len(OPERATION_NAMES)
    assert hysteretic_control_per_use < context_control_per_use
