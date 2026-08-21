from __future__ import annotations

"""Ferroelectric-inspired hysteretic branch state for Gate 0H.

This is a software abstraction, not a material simulation. Each branch owns a
continuous polarization-like state ``p`` in a double-well potential. External
local drive tilts the potential; sufficiently strong/long drive can move the
state across the barrier. After relaxation, ``p≈-1`` or ``p≈+1`` is stable with
zero external drive.

The useful computational distinction is nonvolatile local configuration:
retention does not require a mandatory recurrent transition on every silent
clock tick. A plain explicit latch is therefore a mandatory control; Gate 0H
must not claim unique memory from hysteresis alone.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class HysteresisConfig:
    dt: float = 0.08
    substeps: int = 15
    settle_steps: int = 100
    conductance_gain: float = 4.0


class LandauBranchMatter:
    """Continuous double-well local state.

    Potential:

        U(p; E) = 1/4 (p^2 - 1)^2 - E p

    Gradient-flow dynamics:

        dp/dt = -dU/dp = p - p^3 + E

    The update is differentiable with respect to the applied field sequence.
    """

    def __init__(self, config: HysteresisConfig | None = None) -> None:
        self.config = config or HysteresisConfig()

    def energy(self, p: Tensor, field: Tensor | float = 0.0) -> Tensor:
        e = torch.as_tensor(field, dtype=p.dtype, device=p.device)
        return 0.25 * (p.square() - 1.0).square() - e * p

    def integrate(
        self,
        p: Tensor,
        field: Tensor | float,
        *,
        substeps: int | None = None,
    ) -> Tensor:
        """Integrate one externally driven pulse block."""
        steps = self.config.substeps if substeps is None else int(substeps)
        e = torch.as_tensor(field, dtype=p.dtype, device=p.device)
        for _ in range(steps):
            p = p + self.config.dt * (p - p.pow(3) + e)
        return p

    def run_program(self, p0: Tensor, fields: Tensor) -> tuple[Tensor, Tensor]:
        """Apply a sequence of pulse amplitudes.

        ``fields`` can be ``[time]`` or ``[batch, time]``. ``p0`` is broadcast
        against the batch shape. The returned trace includes the initial state.
        """
        if fields.ndim == 1:
            fields = fields.unsqueeze(0)
            squeeze = True
        elif fields.ndim == 2:
            squeeze = False
        else:
            raise ValueError("fields must have shape [time] or [batch, time]")

        p = p0
        if p.ndim == 0:
            p = p.expand(fields.shape[0])
        elif p.shape[0] != fields.shape[0]:
            p = p.expand(fields.shape[0])
        trace = [p]
        for step in range(fields.shape[1]):
            p = self.integrate(p, fields[:, step])
            trace.append(p)
        stacked = torch.stack(trace, dim=1)
        if squeeze:
            return p.squeeze(0), stacked.squeeze(0)
        return p, stacked

    def relax(self, p: Tensor, *, steps: int | None = None) -> Tensor:
        """Let the state fall into the nearest zero-field well."""
        steps = self.config.settle_steps if steps is None else int(steps)
        return self.integrate(p, 0.0, substeps=steps)

    def retain(self, p: Tensor, silent_steps: int) -> tuple[Tensor, int]:
        """Nonvolatile retention abstraction.

        Once the state has relaxed into a stable well, silent time advances do
        not execute a state transition. The returned integer is executed state
        updates during the silent interval and is intentionally zero.
        """
        if silent_steps < 0:
            raise ValueError("silent_steps must be non-negative")
        return p, 0

    def conductance(self, p: Tensor) -> Tensor:
        """Map local configuration to a branch conductance in (0, 1)."""
        return torch.sigmoid(self.config.conductance_gain * p)


class SchmittLatch:
    """Hard explicit latch control with the same nonvolatile semantics."""

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = float(threshold)

    def run_program(self, state0: Tensor, fields: Tensor) -> tuple[Tensor, Tensor]:
        if fields.ndim == 1:
            fields = fields.unsqueeze(0)
            squeeze = True
        elif fields.ndim == 2:
            squeeze = False
        else:
            raise ValueError("fields must have shape [time] or [batch, time]")
        state = state0
        if state.ndim == 0:
            state = state.expand(fields.shape[0]).clone()
        elif state.shape[0] != fields.shape[0]:
            state = state.expand(fields.shape[0]).clone()
        trace = [state]
        for step in range(fields.shape[1]):
            e = fields[:, step]
            state = torch.where(
                e >= self.threshold,
                torch.ones_like(state),
                torch.where(e <= -self.threshold, -torch.ones_like(state), state),
            )
            trace.append(state)
        stacked = torch.stack(trace, dim=1)
        if squeeze:
            return state.squeeze(0), stacked.squeeze(0)
        return state, stacked

    def retain(self, state: Tensor, silent_steps: int) -> tuple[Tensor, int]:
        if silent_steps < 0:
            raise ValueError("silent_steps must be non-negative")
        return state, 0


class MemorylessGate:
    """Current-drive-only control: no history survives after the field is gone."""

    def __init__(self, gain: float = 4.0) -> None:
        self.gain = float(gain)

    def read(self, field: Tensor | float) -> Tensor:
        e = torch.as_tensor(field, dtype=torch.float32)
        return torch.sigmoid(self.gain * e)
