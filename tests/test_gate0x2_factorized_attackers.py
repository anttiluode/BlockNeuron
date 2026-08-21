from __future__ import annotations

import torch

from blockneuron.factorized_attacker import (
    FactorizedAttackerConfig,
    FactorizedAttackerModel,
)
from experiments.gate0x2_factorized_attacker import X2_REFERENCE_PARAMS
from experiments.gate0x2_receptor_composition import compute_losses, evaluate_compositions


def tiny_model(kind: str) -> FactorizedAttackerModel:
    config = FactorizedAttackerConfig(
        attacker=kind,  # type: ignore[arg-type]
        object_dim=6,
        attribute_dim=4,
        visual_dim=8,
        state_dim=12,
        public_dim=5,
        detail_dim=7,
        basis_dim=10,
        mlp_hidden=24,
        gru_token_dim=8,
        gru_hidden=10,
    )
    return FactorizedAttackerModel(config)


def test_default_attackers_are_near_x2_parameter_budget() -> None:
    for kind in ["mlp", "gru"]:
        model = FactorizedAttackerModel(FactorizedAttackerConfig(attacker=kind))  # type: ignore[arg-type]
        params = sum(p.numel() for p in model.parameters())
        ratio = params / X2_REFERENCE_PARAMS
        assert 0.80 <= ratio <= 1.20, (kind, params, ratio)


def test_attackers_have_no_block_or_phase_machine() -> None:
    for kind in ["mlp", "gru"]:
        model = tiny_model(kind)
        assert not hasattr(model, "block")
        names = [name for name, _ in model.named_parameters()]
        assert not any("phase" in name for name in names)
        assert not any("receptor" in name for name in names)


def test_factorized_attacker_interfaces_and_gradients() -> None:
    for kind in ["mlp", "gru"]:
        torch.manual_seed(4)
        model = tiny_model(kind)
        objects = torch.tensor([0, 1, 2, 3])
        attrs = torch.tensor([0, 1, 2, 3])
        image = torch.rand(4, 1, 28, 28)

        semantic = model.encode(object_ids=objects, attribute_ids=attrs)
        visual = model.encode(image=image)
        paired = model.encode(object_ids=objects, attribute_ids=attrs, image=image)
        assert semantic["public"].shape == (4, 5)
        assert visual["detail"].shape == (4, 7)
        assert paired["state"].shape == (4, 12)
        assert model.decode(semantic["public"]).shape == (4, 1, 28, 28)

        loss = (
            semantic["public"].square().mean()
            + visual["detail"].square().mean()
            + paired["state"].square().mean()
        )
        loss.backward()
        assert model.object_embedding.weight.grad is not None
        assert model.attribute_embedding.weight.grad is not None
        assert model.image_encoder.proj[1].weight.grad is not None


def test_attackers_execute_exact_x2_loss_and_40_way_eval() -> None:
    torch.manual_seed(9)
    for kind in ["mlp", "gru"]:
        model = tiny_model(kind)
        image = torch.rand(4, 1, 28, 28)
        labels = torch.tensor([0, 1, 2, 3])
        attrs = torch.tensor([0, 1, 2, 3])
        prototype = torch.rand(4, 1, 28, 28)

        loss, metrics = compute_losses(
            model,
            image,
            labels,
            attrs,
            attrs,
            prototype,
            phase_mode="dynamic",
        )
        assert torch.isfinite(loss)
        assert set(metrics) == {"loss", "recon", "public", "align", "class", "attr"}
        loss.backward()

        bank = torch.rand(10, 4, 1, 28, 28)
        eval_metrics, grid, public_grid = evaluate_compositions(
            model,
            bank,
            device=torch.device("cpu"),
            phase_mode="dynamic",
        )
        assert grid.shape == (10, 4, 1, 28, 28)
        assert public_grid.shape == (10, 4, 5)
        assert "heldout_visual_nn_joint" in eval_metrics
