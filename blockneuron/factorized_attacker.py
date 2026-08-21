from __future__ import annotations

"""Ordinary factorized attackers for Gate 0X2.

These models keep object and attribute factors separate at the input, exactly as
Gate 0X2 does, but remove BlockNeuron branch gates, phase, and passing modes.
They reuse the same image encoder, public/private split, probes, and coordinate
decoder so the comparison asks whether factor separation itself explains the
held-out compositional rescue.
"""

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .crossmodal import CoordinateImageDecoder, TinyImageEncoder


AttackerKind = Literal["mlp", "gru"]


@dataclass
class FactorizedAttackerConfig:
    image_size: int = 28
    num_classes: int = 10
    num_attributes: int = 4
    object_dim: int = 16
    attribute_dim: int = 8
    visual_dim: int = 48
    state_dim: int = 64
    public_dim: int = 24
    detail_dim: int = 32
    basis_dim: int = 128
    attacker: AttackerKind = "mlp"
    # Chosen so total parameter count is close to the 131,320-parameter X2 block.
    mlp_hidden: int = 512
    gru_token_dim: int = 64
    gru_hidden: int = 112

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


class FactorizedMLPFusion(nn.Module):
    """Plain concatenation + MLP. No recurrence, branch gates, or phase."""

    def __init__(self, config: FactorizedAttackerConfig) -> None:
        super().__init__()
        in_dim = config.object_dim + config.attribute_dim + config.visual_dim + 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, config.mlp_hidden),
            nn.GELU(),
            nn.Linear(config.mlp_hidden, config.state_dim),
            nn.LayerNorm(config.state_dim),
        )

    def forward(
        self,
        object_drive: Tensor,
        attribute_drive: Tensor,
        visual_drive: Tensor,
        mask: Tensor,
        *,
        return_trace: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        x = torch.cat([object_drive, attribute_drive, visual_drive, mask], dim=-1)
        h = self.net(x)
        if return_trace:
            return h, h[:, None, :]
        return h


class FactorizedGRUFusion(nn.Module):
    """Ordinary token-sequence fusion of object, attribute, and image factors."""

    def __init__(self, config: FactorizedAttackerConfig) -> None:
        super().__init__()
        td = config.gru_token_dim
        self.object_proj = nn.Linear(config.object_dim + 1, td)
        self.attribute_proj = nn.Linear(config.attribute_dim + 1, td)
        self.visual_proj = nn.Linear(config.visual_dim + 1, td)
        self.type_embedding = nn.Parameter(torch.randn(3, td) * 0.05)
        self.gru = nn.GRU(td, config.gru_hidden, batch_first=True)
        self.to_state = nn.Linear(config.gru_hidden, config.state_dim)
        self.state_norm = nn.LayerNorm(config.state_dim)

    def forward(
        self,
        object_drive: Tensor,
        attribute_drive: Tensor,
        visual_drive: Tensor,
        mask: Tensor,
        *,
        return_trace: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        obj = self.object_proj(torch.cat([object_drive, mask[:, 0:1]], dim=-1))
        attr = self.attribute_proj(torch.cat([attribute_drive, mask[:, 1:2]], dim=-1))
        vis = self.visual_proj(torch.cat([visual_drive, mask[:, 2:3]], dim=-1))
        tokens = torch.stack([obj, attr, vis], dim=1) + self.type_embedding[None]
        seq, hidden = self.gru(tokens)
        h = self.state_norm(self.to_state(hidden[-1]))
        if return_trace:
            trace_states = self.state_norm(self.to_state(seq))
            return h, trace_states
        return h


class FactorizedAttackerModel(nn.Module):
    """Matched-interface ordinary baseline for Gate 0X2.

    Object/attribute IDs remain factorized at entry, but there is no BlockNeuron
    machinery. This class intentionally mirrors ReceptorCompositionModel.encode
    so Gate 0X2's loss and evaluation code can be reused unchanged.
    """

    def __init__(self, config: FactorizedAttackerConfig) -> None:
        super().__init__()
        if config.attacker not in {"mlp", "gru"}:
            raise ValueError("attacker must be 'mlp' or 'gru'")
        self.config = config
        self.object_embedding = nn.Embedding(config.num_classes, config.object_dim)
        self.attribute_embedding = nn.Embedding(config.num_attributes, config.attribute_dim)
        self.image_encoder = TinyImageEncoder(config.visual_dim)
        if config.attacker == "mlp":
            self.fusion: nn.Module = FactorizedMLPFusion(config)
        else:
            self.fusion = FactorizedGRUFusion(config)

        self.public_head = nn.Sequential(
            nn.Linear(config.state_dim, config.public_dim),
            nn.LayerNorm(config.public_dim),
        )
        self.detail_head = nn.Linear(config.state_dim, config.detail_dim)
        self.classifier = nn.Linear(config.public_dim, config.num_classes)
        self.attribute_classifier = nn.Linear(config.public_dim, config.num_attributes)
        self.decoder = CoordinateImageDecoder(
            config.image_size,
            config.public_dim + config.detail_dim,
            config.basis_dim,
        )

    def encode(
        self,
        *,
        object_ids: Tensor | None = None,
        attribute_ids: Tensor | None = None,
        image: Tensor | None = None,
        phase_mode: str = "dynamic",
        return_trace: bool = False,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        # phase_mode is accepted only for interface parity; the attacker has no phase.
        del phase_mode
        if object_ids is None and attribute_ids is None and image is None:
            raise ValueError("at least one receptor or image modality is required")

        reference = object_ids if object_ids is not None else attribute_ids
        if reference is None:
            assert image is not None
            batch = image.shape[0]
            device = image.device
        else:
            batch = reference.shape[0]
            device = reference.device

        if object_ids is not None:
            if object_ids.shape[0] != batch:
                raise ValueError("object_ids batch mismatch")
            object_drive = self.object_embedding(object_ids)
            object_present = torch.ones(batch, 1, device=device)
        else:
            object_drive = torch.zeros(batch, self.config.object_dim, device=device)
            object_present = torch.zeros(batch, 1, device=device)

        if attribute_ids is not None:
            if attribute_ids.shape[0] != batch:
                raise ValueError("attribute_ids batch mismatch")
            attribute_drive = self.attribute_embedding(attribute_ids)
            attribute_present = torch.ones(batch, 1, device=device)
        else:
            attribute_drive = torch.zeros(batch, self.config.attribute_dim, device=device)
            attribute_present = torch.zeros(batch, 1, device=device)

        if image is not None:
            if image.shape[0] != batch:
                raise ValueError("image batch mismatch")
            visual_drive = self.image_encoder(image)
            image_present = torch.ones(batch, 1, device=device)
        else:
            visual_drive = torch.zeros(batch, self.config.visual_dim, device=device)
            image_present = torch.zeros(batch, 1, device=device)

        mask = torch.cat([object_present, attribute_present, image_present], dim=-1)
        fusion_out = self.fusion(
            object_drive,
            attribute_drive,
            visual_drive,
            mask,
            return_trace=return_trace,
        )
        if return_trace:
            h, states = fusion_out
        else:
            h = fusion_out
            states = None

        public = F.normalize(self.public_head(h), dim=-1, eps=1e-6)
        detail = self.detail_head(h)
        result: dict[str, Tensor | dict[str, Tensor]] = {
            "state": h,
            "public": public,
            "detail": detail,
            "logits": self.classifier(public),
            "attr_logits": self.attribute_classifier(public),
        }
        if states is not None:
            result["trace"] = {
                "states": states,
                "gates": torch.empty(batch, states.shape[1], 0, device=device),
                "phases": torch.empty(states.shape[1], 0, device=device),
            }
        return result

    def decode(self, public: Tensor, detail: Tensor | None = None) -> Tensor:
        if detail is None:
            detail = torch.zeros(public.shape[0], self.config.detail_dim, device=public.device)
        return self.decoder(torch.cat([public, detail], dim=-1))

    def decode_state(self, state: Tensor, *, keep_detail: bool = True) -> Tensor:
        public = F.normalize(self.public_head(state), dim=-1, eps=1e-6)
        detail = self.detail_head(state) if keep_detail else None
        return self.decode(public, detail)
