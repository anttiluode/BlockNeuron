import torch

from experiments.gate0x1_fashion_compositional import (
    ATTRIBUTES,
    HOLDOUT_ATTR,
    LABELS,
    assert_split_is_compositional,
    build_prototype_bank,
    heldout_pairs,
    seen_pairs,
    transform_tensor,
)


def _centroid_x(image: torch.Tensor) -> float:
    weights = image[0].sum(dim=0)
    coords = torch.arange(image.shape[-1], dtype=image.dtype)
    return float((weights * coords).sum() / weights.sum().clamp_min(1e-8))


def test_holdout_split_reuses_every_class_and_attribute() -> None:
    assert_split_is_compositional()
    held = set(heldout_pairs())
    seen = set(seen_pairs())
    assert len(held) == len(LABELS)
    assert len(seen) == len(LABELS) * (len(ATTRIBUTES) - 1)
    assert held.isdisjoint(seen)
    for label in range(len(LABELS)):
        assert (label, HOLDOUT_ATTR[label]) in held


def test_controlled_transforms_have_expected_direction() -> None:
    image = torch.zeros(1, 28, 28)
    image[:, 8:20, 9:19] = 1.0

    small = transform_tensor(image, "small")
    large = transform_tensor(image, "large")
    left = transform_tensor(image, "left")
    right = transform_tensor(image, "right")

    assert small.shape == image.shape == large.shape == left.shape == right.shape
    assert small.sum() < image.sum()
    assert large.sum() > image.sum()
    assert _centroid_x(left) < _centroid_x(image) < _centroid_x(right)


def test_prototype_bank_contains_all_combinations() -> None:
    means = torch.rand(len(LABELS), 1, 28, 28)
    bank = build_prototype_bank(means)
    assert bank.shape == (len(LABELS), len(ATTRIBUTES), 1, 28, 28)
    assert torch.isfinite(bank).all()
