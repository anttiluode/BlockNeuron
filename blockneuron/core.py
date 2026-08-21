"""Minimal BlockNeuron primitives.

The design goal is not a biological simulation. A BlockNeuron is one level richer
than a point neuron: slow structural weights are modulated by local synapse state,
a low-dimensional mode signal, and rhythmic phase before currents are integrated
inside dendritic compartments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


def mode_gain(receptor_projection: Tensor, beta: float = 4.0) -> Tensor:
    """Positive branch/edge gain from a local receptor projection."""
    return 2.0 * torch.sigmoid(beta * receptor_projection)


def phase_gain(delta_phase: Tensor, preferred_phase: Tensor, amplitude: float = 0.9) -> Tensor:
    """Rhythmic conductance factor: 1 + a cos(delta_phi - psi)."""
    if not 0.0 <= amplitude < 1.0:
        raise ValueError("amplitude must satisfy 0 <= amplitude < 1")
    return 1.0 + amplitude * torch.cos(delta_phase - preferred_phase)


def effective_conductance(
    weight: Tensor,
    synapse_state: Tensor,
    receptor_projection: Tensor,
    delta_phase: Tensor,
    preferred_phase: Tensor,
    *,
    mode_beta: float = 4.0,
    phase_amplitude: float = 0.9,
) -> Tensor:
    """Replacement for a static scalar synaptic weight.

    g_e(t) = w_e * S(z_e) * M(r^T m) * R(delta_phi, psi_e)
    """
    return (
        weight
        * synapse_state
        * mode_gain(receptor_projection, beta=mode_beta)
        * phase_gain(delta_phase, preferred_phase, amplitude=phase_amplitude)
    )


@dataclass(frozen=True)
class EdgeSpec:
    """Sparse structural connection metadata."""
    source: int
    target: int
    target_branch: int
    delay: int = 0


class BlockNeuronLayer(nn.Module):
    """A small differentiable compartmental unit for experiments.

    Each neuron owns B branches. Inputs first enter branches, branch currents are
    gated by an abstract K-dimensional modulatory vector and a scalar phase, then
    integrated at the soma. This is intentionally a minimal block, not a full
    spiking or cable model.
    """

    def __init__(
        self,
        input_dim: int,
        neurons: int,
        branches: int,
        mod_channels: int,
        *,
        mode_beta: float = 4.0,
        phase_beta: float = 4.0,
    ) -> None:
        super().__init__()
        if min(input_dim, neurons, branches, mod_channels) <= 0:
            raise ValueError("all dimensions must be positive")

        self.input_dim = input_dim
        self.neurons = neurons
        self.branches = branches
        self.mod_channels = mod_channels
        self.mode_beta = mode_beta
        self.phase_beta = phase_beta

        # Slow structural mass: input contacts on each branch and branch->soma cable.
        self.weight = nn.Parameter(torch.empty(neurons, branches, input_dim))
        self.cable = nn.Parameter(torch.ones(neurons, branches))
        self.bias = nn.Parameter(torch.zeros(neurons, branches))

        # Slowly learned/fixed identity-like branch properties.
        self.receptors = nn.Parameter(torch.empty(neurons, branches, mod_channels))
        self.phase_preference = nn.Parameter(torch.empty(neurons, branches))

        nn.init.xavier_uniform_(self.weight)
        nn.init.normal_(self.receptors, std=0.5)
        nn.init.uniform_(self.phase_preference, -math.pi, math.pi)

    def branch_gates(self, modulator: Tensor, phase: Tensor) -> Tensor:
        if modulator.ndim != 2 or modulator.shape[1] != self.mod_channels:
            raise ValueError("modulator must be [batch, mod_channels]")
        if phase.ndim != 1 or phase.shape[0] != modulator.shape[0]:
            raise ValueError("phase must be [batch]")

        # Local receptor projection r_ib^T m.
        chem = (modulator[:, None, None, :] * self.receptors[None, :, :, :]).sum(-1)
        chem_gate = torch.sigmoid(self.mode_beta * chem)

        phase_match = torch.cos(phase[:, None, None] - self.phase_preference[None, :, :])
        rhythm_gate = torch.sigmoid(self.phase_beta * phase_match)
        return chem_gate * rhythm_gate

    def effective_input_matrix(self, modulator: Tensor, phase: Tensor) -> Tensor:
        """Return W_eff for each sample: [batch, neurons, input_dim]."""
        gates = self.branch_gates(modulator, phase)
        branch_to_input = self.cable[:, :, None] * self.weight
        return (gates[:, :, :, None] * branch_to_input[None, :, :, :]).sum(dim=2)

    def forward(self, x: Tensor, modulator: Tensor, phase: Tensor) -> Tensor:
        gates = self.branch_gates(modulator, phase)
        drive = torch.einsum("bi,nri->bnr", x, self.weight) + self.bias[None, :, :]
        dendrite = torch.tanh(drive)
        soma = (gates * self.cable[None, :, :] * dendrite).sum(dim=-1)
        return soma


class FourModeBlock(nn.Module):
    """Gate-0 instrument: four branches with fixed mode identities.

    Chemistry selects one branch pair; phase selects one branch inside that pair.
    Only structural branch weights and a scalar bias are trained in Gate 0.
    """

    def __init__(self, input_dim: int, gate_beta: float = 7.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.gate_beta = gate_beta
        self.weight = nn.Parameter(torch.empty(4, input_dim))
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.weight, std=0.2)

        self.register_buffer("chem_preference", torch.tensor([-1.0, -1.0, 1.0, 1.0]))
        self.register_buffer("phase_preference", torch.tensor([0.0, math.pi, 0.0, math.pi]))

    def gates(self, chemical_mode: Tensor, phase: Tensor) -> Tensor:
        chem = torch.sigmoid(
            self.gate_beta * chemical_mode[:, None] * self.chem_preference[None, :]
        )
        rhythm = torch.sigmoid(
            self.gate_beta * torch.cos(phase[:, None] - self.phase_preference[None, :])
        )
        return chem * rhythm

    def effective_weight(self, chemical_mode: Tensor, phase: Tensor) -> Tensor:
        return self.gates(chemical_mode, phase) @ self.weight

    def forward(self, x: Tensor, chemical_mode: Tensor, phase: Tensor) -> Tensor:
        return (self.effective_weight(chemical_mode, phase) * x).sum(-1) + self.bias


class HyperLinearAttacker(nn.Module):
    """Strong ordinary control: context directly generates the effective weight."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.coeff = nn.Parameter(torch.empty(4, input_dim))
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.coeff, std=0.2)

    def effective_weight(self, chemical_mode: Tensor, phase: Tensor) -> Tensor:
        p = torch.cos(phase)
        basis = torch.stack(
            [torch.ones_like(chemical_mode), chemical_mode, p, chemical_mode * p], dim=-1
        )
        return basis @ self.coeff

    def forward(self, x: Tensor, chemical_mode: Tensor, phase: Tensor) -> Tensor:
        return (self.effective_weight(chemical_mode, phase) * x).sum(-1) + self.bias


class UnconditionedLinear(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: Tensor, chemical_mode: Tensor, phase: Tensor) -> Tensor:
        del chemical_mode, phase
        return self.linear(x).squeeze(-1)
