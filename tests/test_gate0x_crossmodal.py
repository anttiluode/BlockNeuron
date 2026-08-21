import torch

from blockneuron.crossmodal import CharTokenizer, CrossModalBlockModel, CrossModalConfig


def tiny_model() -> tuple[CrossModalBlockModel, CharTokenizer]:
    config = CrossModalConfig(
        semantic_dim=16,
        visual_dim=16,
        state_dim=24,
        public_dim=8,
        detail_dim=8,
        branches=4,
        phase_dim=3,
        steps=4,
        basis_dim=24,
    )
    return CrossModalBlockModel(config), CharTokenizer(max_len=config.max_text_len)


def test_crossmodal_operating_modes_have_expected_shapes():
    model, tokenizer = tiny_model()
    tokens = tokenizer.encode(["bag", "ankle boot"])
    image = torch.rand(2, 1, 28, 28)

    for kwargs in ({"tokens": tokens}, {"image": image}, {"tokens": tokens, "image": image}):
        out = model.encode(**kwargs)
        assert out["state"].shape == (2, 24)
        assert out["public"].shape == (2, 8)
        assert out["detail"].shape == (2, 8)
        assert out["logits"].shape == (2, 10)
        decoded = model.decode(out["public"], out["detail"])
        assert decoded.shape == (2, 1, 28, 28)
        assert torch.isfinite(decoded).all()


def test_dynamic_phase_moves_gates_while_clamped_phase_repeats_them():
    model, tokenizer = tiny_model()
    tokens = tokenizer.encode(["sneaker"])

    dynamic = model.encode(tokens=tokens, phase_mode="dynamic", return_trace=True)["trace"]
    clamped = model.encode(tokens=tokens, phase_mode="clamped", return_trace=True)["trace"]

    dynamic_gates = dynamic["gates"]
    clamped_gates = clamped["gates"]
    assert not torch.allclose(dynamic_gates[:, 0], dynamic_gates[:, -1])
    assert torch.allclose(clamped_gates[:, 0], clamped_gates[:, -1], atol=1e-7)


def test_text_public_state_can_decode_with_private_detail_removed():
    model, tokenizer = tiny_model()
    tokens = tokenizer.encode(["dress", "coat"])
    out = model.encode(tokens=tokens)
    image = model.decode(out["public"], None)
    assert image.shape == (2, 1, 28, 28)
