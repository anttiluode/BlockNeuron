from __future__ import annotations

from experiments.gate0x2_replication_suite import aggregate, parse_int_list, parse_str_list


def fake_row(model: str, seed: int, offset: float) -> dict[str, object]:
    row: dict[str, object] = {
        "model": model,
        "seed": seed,
        "parameters": 131_000,
        "final_epoch": 16,
        "selected_epoch": 14,
    }
    metrics = [
        "seen_joint_acc",
        "heldout_joint_acc",
        "heldout_class_acc",
        "heldout_attr_acc",
        "seen_proto_mse",
        "heldout_proto_mse",
        "seen_visual_nn_joint",
        "heldout_visual_nn_joint",
    ]
    for idx, metric in enumerate(metrics):
        value = offset + idx * 0.01
        row[f"final_{metric}"] = value
        row[f"selected_{metric}"] = value + 0.005
    return row


def test_replication_argument_parsers() -> None:
    assert parse_int_list("18001, 18002") == [18001, 18002]
    assert parse_str_list("x2,mlp,gru") == ["x2", "mlp", "gru"]


def test_replication_aggregate_mean_and_std() -> None:
    rows = [
        fake_row("x2", 1, 0.1),
        fake_row("x2", 2, 0.2),
        fake_row("mlp", 1, 0.3),
    ]
    summary = aggregate(rows)
    assert set(summary) == {"x2", "mlp"}
    assert abs(summary["x2"]["final"]["seen_joint_acc_mean"] - 0.15) < 1e-9
    assert summary["x2"]["final"]["seen_joint_acc_std"] > 0
    assert summary["mlp"]["final"]["seen_joint_acc_std"] == 0.0
    assert summary["x2"]["parameters"]["mean"] == 131_000
