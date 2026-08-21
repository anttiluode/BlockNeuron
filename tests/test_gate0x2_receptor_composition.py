from __future__ import annotations

import torch

from blockneuron.receptor_composition import (
    ReceptorCompositionConfig,
    ReceptorCompositionModel,
)
from experiments.gate0x1_fashion_compositional import (
    ATTRIBUTES,
    HOLDOUT_ATTR,
    LABELS,
    heldout_pairs,
    seen_pairs,
)
from experiments.gate0x2_receptor_composition import (
    compute_losses,
    evaluate_compositions,
)


def tiny_model() -> ReceptorCompositionModel:
    config = ReceptorCompositionConfig(
        object_dim=6,
        attribute_dim=4,
        visual_dim=8,
        state_dim=12,
        public_dim=5,
        detail_dim=7,
        branches=4,
        phase_dim=2,
        steps=3,
        basis_dim=10,
    )
    return ReceptorCompositionModel(config)


def test_receptor_model_keeps_factor_families_separate() -> None:
    model = tiny_model()
    assert not hasattr(model, "text_encoder")
    assert model.block.object_receptors.shape == (4, 6)
    assert model.block.attribute_receptors.shape == (4, 4)


def test_receptor_and_image_paths_share_outputs_and_gradients() -> None:
    torch.manual_seed(3)
    model = tiny_model()
    objects = torch.tensor([0, 1, 2])
    attrs = torch.tensor([0, 1, 2])
    image = torch.rand(3, 1, 28, 28)

    receptor = model.encode(object_ids=objects, attribute_ids=attrs)
    image_only = model.encode(image=image)
    paired = model.encode(
        object_ids=objects, attribute_ids=attrs, image=image
    )

    assert receptor["public"].shape == (3, 5)
    assert image_only["public"].shape == (3, 5)
    assert paired["state"].shape == (3, 12)
    assert model.decode(receptor["public"]).shape == (3, 1, 28, 28)

    loss = (
        receptor["public"].square().mean()
        + image_only["detail"].square().mean()
        + paired["state"].square().mean()
    )
    loss.backward()
    assert model.object_embedding.weight.grad is not None
    assert model.attribute_embedding.weight.grad is not None
    assert model.image_encoder.proj[1].weight.grad is not None
    assert model.block.object_receptors.grad is not None
    assert model.block.attribute_receptors.grad is not None


def test_object_and_attribute_changes_both_reach_block() -> None:
    torch.manual_seed(7)
    model = tiny_model()
    same_object = torch.tensor([2, 2])
    different_attrs = torch.tensor([0, 1])
    out_attr = model.encode(
        object_ids=same_object, attribute_ids=different_attrs
    )["state"]
    assert not torch.allclose(out_attr[0], out_attr[1])

    different_objects = torch.tensor([2, 3])
    same_attr = torch.tensor([1, 1])
    out_object = model.encode(
        object_ids=different_objects, attribute_ids=same_attr
    )["state"]
    assert not torch.allclose(out_object[0], out_object[1])


def test_x2_uses_same_strict_compositional_split_as_x1() -> None:
    held = set(heldout_pairs())
    seen = set(seen_pairs())
    assert len(held) == len(LABELS) == 10
    assert len(seen) == len(LABELS) * (len(ATTRIBUTES) - 1) == 30
    assert held.isdisjoint(seen)

    for label in range(len(LABELS)):
        assert (label, HOLDOUT_ATTR[label]) in held
        assert all(
            (label, attr) in seen
            for attr in range(len(ATTRIBUTES))
            if attr != HOLDOUT_ATTR[label]
        )


def test_x2_loss_and_all_40_composition_eval_are_executable() -> None:
    torch.manual_seed(11)
    model = tiny_model()
    image = torch.rand(4, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3])
    visual_attrs = torch.tensor([1, 2, 3, 0])
    receptor_attrs = visual_attrs.clone()
    prototype = torch.rand(4, 1, 28, 28)

    loss, metrics = compute_losses(
        model,
        image,
        labels,
        visual_attrs,
        receptor_attrs,
        prototype,
        phase_mode="dynamic",
    )
    assert torch.isfinite(loss)
    assert set(metrics) == {"loss", "recon", "public", "align", "class", "attr"}
    loss.backward()
    assert model.block.input_weight.grad is not None

    prototype_bank = torch.rand(10, 4, 1, 28, 28)
    eval_metrics, grid, public_grid = evaluate_compositions(
        model,
        prototype_bank,
        device=torch.device("cpu"),
        phase_mode="dynamic",
    )
    assert grid.shape == (10, 4, 1, 28, 28)
    assert public_grid.shape == (10, 4, 5)
    assert "heldout_visual_nn_joint" in eval_metrics
