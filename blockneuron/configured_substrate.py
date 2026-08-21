from __future__ import annotations

"""Gate 0H1 — write-once configured computation.

This module composes the continuous hysteretic state from :mod:`hysteresis`
with a small bank of deterministic linear operations. A short programming
field selects which route is physically available; after relaxation the
program can disappear and subsequent data reads do not update the material
state.

The module deliberately includes ordinary controls. An explicit latch has the
same write-once/hold capability, so H1 is a computational-organization test,
not a unique-memory or unique-expressivity claim.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from .hysteresis import HysteresisConfig, LandauBranchMatter


OPERATION_NAMES = ("copy", "negate", "rotate", "filter")


def operation_matrices(*, dtype: torch.dtype = torch.float32, device=None) -> Tensor:
    """Return four fixed 4x4 operations used by Gate 0H1."""
    eye = torch.eye(4, dtype=dtype, device=device)
    negate = -eye
    rotate = torch.tensor(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    filt = torch.tensor(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5],
        ],
        dtype=dtype,
        device=device,
    )
    return torch.stack([eye, negate, rotate, filt], dim=0)


def apply_matrix_bank(x: Tensor, matrices: Tensor, operation_id: int) -> Tensor:
    """Apply one operation from ``matrices`` to a batch of row vectors."""
    if x.shape[-1] != matrices.shape[-1]:
        raise ValueError("input dimension must match operation matrices")
    op = int(operation_id)
    if op < 0 or op >= matrices.shape[0]:
        raise ValueError("operation_id out of range")
    return x @ matrices[op].transpose(-1, -2)


@dataclass
class ProgramReceipt:
    field: Tensor
    before: Tensor
    after_drive: Tensor
    settled: Tensor
    selected_operation: int
    switch_work_proxy: float
    external_control_values: int
    internal_scalar_updates: int


class HystereticOperationSubstrate:
    """A four-route substrate configured by a persistent double-well state."""

    def __init__(
        self,
        *,
        write_amplitude: float = 1.2,
        erase_amplitude: float = 1.2,
        config: HysteresisConfig | None = None,
    ) -> None:
        if config is None:
            # Gain 8 makes the relaxed +/-1 wells almost one-hot as conductances
            # while leaving a smooth readout available for diagnostics.
            config = HysteresisConfig(conductance_gain=8.0)
        self.material = LandauBranchMatter(config)
        self.write_amplitude = float(write_amplitude)
        self.erase_amplitude = float(erase_amplitude)
        self.matrices = operation_matrices()

    @property
    def num_operations(self) -> int:
        return int(self.matrices.shape[0])

    def initial_state(self, *, device=None, dtype: torch.dtype = torch.float32) -> Tensor:
        return -torch.ones(self.num_operations, device=device, dtype=dtype)

    def program(
        self,
        state: Tensor,
        operation_id: int,
        *,
        noise_std: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, ProgramReceipt]:
        """Write one route, erase the others, then relax into stable wells."""
        op = int(operation_id)
        if op < 0 or op >= self.num_operations:
            raise ValueError("operation_id out of range")
        if state.shape != (self.num_operations,):
            raise ValueError("state must have one scalar per operation")

        field = torch.full_like(state, -self.erase_amplitude)
        field[op] = self.write_amplitude
        if noise_std > 0.0:
            noise = torch.randn(
                field.shape,
                dtype=field.dtype,
                device=field.device,
                generator=generator,
            )
            field = field + float(noise_std) * noise

        before = state.clone()
        after_drive = self.material.integrate(before, field)
        settled = self.material.relax(after_drive)
        selected = self.selected_operation(settled)
        switch_work = float((settled - before).abs().sum().detach())
        cfg = self.material.config
        internal_updates = self.num_operations * (cfg.substeps + cfg.settle_steps)
        receipt = ProgramReceipt(
            field=field.detach().clone(),
            before=before.detach().clone(),
            after_drive=after_drive.detach().clone(),
            settled=settled.detach().clone(),
            selected_operation=selected,
            switch_work_proxy=switch_work,
            external_control_values=self.num_operations,
            internal_scalar_updates=internal_updates,
        )
        return settled, receipt

    def retain(self, state: Tensor, silent_steps: int) -> tuple[Tensor, int]:
        return self.material.retain(state, silent_steps)

    def conductance(self, state: Tensor) -> Tensor:
        return self.material.conductance(state)

    def selected_operation(self, state: Tensor) -> int:
        return int(self.conductance(state).argmax().item())

    def apply(self, x: Tensor, state: Tensor) -> Tensor:
        """Hard routed read: only the configured operation is executed."""
        matrices = self.matrices.to(dtype=x.dtype, device=x.device)
        return apply_matrix_bank(x, matrices, self.selected_operation(state))

    def apply_soft(self, x: Tensor, state: Tensor) -> Tensor:
        """Differentiable diagnostic read using all conductances as mixture weights."""
        matrices = self.matrices.to(dtype=x.dtype, device=x.device)
        g = self.conductance(state.to(dtype=x.dtype, device=x.device))
        weights = g / g.sum().clamp_min(1e-12)
        all_y = torch.einsum("nd,odk->nok", x, matrices.transpose(-1, -2))
        return (weights[None, :, None] * all_y).sum(dim=1)


class ExplicitOperationLatch:
    """Mandatory ordinary control: exact write-once nonvolatile operation latch."""

    def __init__(self) -> None:
        self.matrices = operation_matrices()

    def initial_state(self) -> Tensor:
        return torch.zeros(len(OPERATION_NAMES))

    def program(self, operation_id: int) -> Tensor:
        return F.one_hot(
            torch.tensor(int(operation_id)), num_classes=len(OPERATION_NAMES)
        ).to(torch.float32)

    def retain(self, state: Tensor, silent_steps: int) -> tuple[Tensor, int]:
        if silent_steps < 0:
            raise ValueError("silent_steps must be non-negative")
        return state, 0

    def apply(self, x: Tensor, state: Tensor) -> Tensor:
        op = int(state.argmax().item())
        matrices = self.matrices.to(dtype=x.dtype, device=x.device)
        return apply_matrix_bank(x, matrices, op)


class ContextOperationMux:
    """Memoryless control: operation context must be supplied on every read."""

    def __init__(self) -> None:
        self.matrices = operation_matrices()

    def apply(self, x: Tensor, operation_id: int) -> Tensor:
        matrices = self.matrices.to(dtype=x.dtype, device=x.device)
        return apply_matrix_bank(x, matrices, operation_id)


class ClockedOperationKeeper:
    """Clocked state baseline.

    It stores an exact one-hot operation state, but a conventional clocked read
    executes an identity state transition before using the state. This is only
    an accounting contrast. An event-driven implementation can skip that update
    and then reduces to the explicit latch control above.
    """

    def __init__(self) -> None:
        self.matrices = operation_matrices()

    def program(self, operation_id: int) -> Tensor:
        return F.one_hot(
            torch.tensor(int(operation_id)), num_classes=len(OPERATION_NAMES)
        ).to(torch.float32)

    def read(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor, int]:
        next_state = state.clone()  # explicit clocked state transition
        op = int(next_state.argmax().item())
        matrices = self.matrices.to(dtype=x.dtype, device=x.device)
        return apply_matrix_bank(x, matrices, op), next_state, 1
