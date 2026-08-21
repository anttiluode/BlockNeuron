import math

import torch

from blockneuron import BlockNeuronLayer, FourModeBlock, effective_conductance


def test_effective_conductance_changes_with_mode_and_phase():
    w = torch.tensor(1.0)
    z = torch.tensor(1.0)
    psi = torch.tensor(0.0)
    favored = effective_conductance(w, z, torch.tensor(1.0), torch.tensor(0.0), psi)
    suppressed = effective_conductance(w, z, torch.tensor(-1.0), torch.tensor(math.pi), psi)
    assert favored > suppressed * 20


def test_four_contexts_select_four_different_branches():
    model = FourModeBlock(8, gate_beta=10.0)
    chem = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    phase = torch.tensor([0.0, math.pi, 0.0, math.pi])
    gates = model.gates(chem, phase)
    winners = gates.argmax(dim=-1)
    assert winners.tolist() == [0, 1, 2, 3]
    assert torch.all(gates.max(dim=-1).values > 0.98)


def test_hand_set_structure_realizes_four_rules():
    torch.manual_seed(2)
    model = FourModeBlock(8, gate_beta=12.0)
    h = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1],
            [1, -1, -1, 1, 1, -1, -1, 1],
        ],
        dtype=torch.float32,
    ) / math.sqrt(8)
    with torch.no_grad():
        model.weight.copy_(h)
        model.bias.zero_()

    n = 12000
    x = torch.randn(n, 8)
    ci = torch.randint(0, 2, (n,))
    pi = torch.randint(0, 2, (n,))
    chem = ci.float() * 2 - 1
    phase = pi.float() * math.pi
    idx = ci * 2 + pi
    y = (x * h[idx]).sum(-1) > 0
    pred = model(x, chem, phase) > 0
    assert (pred == y).float().mean() > 0.995


def test_general_block_layer_shapes_and_effective_matrix():
    torch.manual_seed(3)
    layer = BlockNeuronLayer(input_dim=6, neurons=3, branches=4, mod_channels=2)
    x = torch.randn(5, 6)
    m = torch.randn(5, 2)
    phase = torch.linspace(0, math.pi, 5)
    y = layer(x, m, phase)
    w_eff = layer.effective_input_matrix(m, phase)
    assert y.shape == (5, 3)
    assert w_eff.shape == (5, 3, 6)
    assert torch.isfinite(y).all()
