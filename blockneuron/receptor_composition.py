from __future__ import annotations

"""Factor-separated semantic receptors for Gate 0X2.

Unlike Gate 0X1, object identity and visual quality are not collapsed into a
single text embedding before the shared block. They arrive as two distinct
receptor populations and only meet inside the recurrent branch bank.
"""

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .crossmodal import CoordinateImageDecoder, TinyImageEncoder


@dataclass
class ReceptorCompositionConfig:
    image_size: int = 28
    num_classes: int = 10
    num_attributes: int = 4
    object_dim: int = 16
    attribute_dim: int = 8
    visual_dim: int = 48
    state_dim: int = 64
    public_dim: int = 24
    detail_dim: int = 32
    branches: int = 8
    phase_dim: int = 4
    steps: int = 8
    basis_dim: int = 128

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class FactorSeparatedReceptorBlock(nn.Module):
    """One recurrent block with separate object, quality, and visual receptors."""

    def __init__(
        self,
        object_dim: int,
        attribute_dim: int,
        visual_dim: int,
        state_dim: int,
        branches: int,
        phase_dim: int,
        steps: int,
    ) -> None:
        super().__init__()
        self.object_dim = object_dim
        self.attribute_dim = attribute_dim
        self.visual_dim = visual_dim
        self.state_dim = state_dim
        self.branches = branches
        self.phase_dim = phase_dim
        self.steps = steps

        # Three independent receptor families. Their scores are not fused outside
        # the block; branch gates are where the factors first interact.
        self.object_receptors = nn.Parameter(torch.randn(branches, object_dim) * 0.25)
        self.attribute_receptors = nn.Parameter(torch.randn(branches, attribute_dim) * 0.25)
        self.visual_receptors = nn.Parameter(torch.randn(branches, visual_dim) * 0.25)
        self.mask_receptors = nn.Parameter(torch.randn(branches, 3) * 0.1)

        drive_dim = object_dim + attribute_dim + visual_dim + 3
        self.input_weight = nn.Parameter(torch.empty(branches, state_dim, drive_dim))
        self.recurrent_weight = nn.Parameter(torch.empty(branches, state_dim, state_dim))
        self.branch_bias = nn.Parameter(torch.zeros(branches, state_dim))
        self.cable = nn.Parameter(torch.ones(branches))

        self.phase_preference = nn.Parameter(torch.empty(branches, phase_dim))
        self.phase0 = nn.Parameter(torch.zeros(phase_dim))
        base_omega = torch.tensor([0.53, 0.79, 1.11, 1.41])
        if phase_dim <= len(base_omega):
            omega = base_omega[:phase_dim].clone()
        else:
            omega = torch.cat(
                [base_omega, torch.linspace(1.57, 2.17, phase_dim - 4)], dim=0
            )
        self.omega = nn.Parameter(omega)

        self.update_logit = nn.Parameter(torch.tensor(-0.45))
        self.state_norm = nn.LayerNorm(state_dim)

        nn.init.xavier_uniform_(self.input_weight)
        for branch in range(branches):
            nn.init.orthogonal_(self.recurrent_weight[branch])
        nn.init.uniform_(self.phase_preference, -math.pi, math.pi)

    def _content_score(
        self,
        object_drive: Tensor,
        attribute_drive: Tensor,
        visual_drive: Tensor,
        mask: Tensor,
    ) -> Tensor:
        obj = F.normalize(object_drive, dim=-1, eps=1e-6)
        attr = F.normalize(attribute_drive, dim=-1, eps=1e-6)
        vis = F.normalize(visual_drive, dim=-1, eps=1e-6)
        obj_r = F.normalize(self.object_receptors, dim=-1, eps=1e-6)
        attr_r = F.normalize(self.attribute_receptors, dim=-1, eps=1e-6)
        vis_r = F.normalize(self.visual_receptors, dim=-1, eps=1e-6)
        return (
            obj @ obj_r.T
            + attr @ attr_r.T
            + vis @ vis_r.T
            + mask @ self.mask_receptors.T
        )

    def forward(
        self,
        object_drive: Tensor,
        attribute_drive: Tensor,
        visual_drive: Tensor,
        mask: Tensor,
        *,
        phase_mode: str = "dynamic",
        return_trace: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if phase_mode not in {"dynamic", "clamped"}:
            raise ValueError("phase_mode must be 'dynamic' or 'clamped'")
        batch = object_drive.shape[0]
        if (
            attribute_drive.shape[0] != batch
            or visual_drive.shape[0] != batch
            or mask.shape[0] != batch
        ):
            raise ValueError("all drives must share the same batch dimension")

        h = object_drive.new_zeros(batch, self.state_dim)
        content_score = self._content_score(
            object_drive, attribute_drive, visual_drive, mask
        )
        drive = torch.cat(
            [object_drive, attribute_drive, visual_drive, mask], dim=-1
        )
        input_drive = (
            torch.einsum("nm,rdm->nrd", drive, self.input_weight)
            + self.branch_bias
        )
        alpha = torch.sigmoid(self.update_logit)

        states: list[Tensor] = []
        gates_trace: list[Tensor] = []
        phases: list[Tensor] = []

        for step in range(self.steps):
            if phase_mode == "dynamic":
                phase = self.phase0 + float(step) * self.omega
            else:
                phase = torch.zeros_like(self.phase0)
            phase_score = torch.cos(
                phase[None, None, :] - self.phase_preference[None, :, :]
            ).mean(-1)
            gates = torch.softmax(2.0 * content_score + 2.0 * phase_score, dim=-1)

            recurrent = torch.einsum(
                "ni,rdi->nrd", h, self.recurrent_weight
            )
            candidate = torch.tanh(input_drive + recurrent)
            cable = torch.sigmoid(self.cable)[None, :, None] * 2.0
            mixed = (gates[:, :, None] * cable * candidate).sum(dim=1)
            h = self.state_norm((1.0 - alpha) * h + alpha * mixed)

            if return_trace:
                states.append(h)
                gates_trace.append(gates)
                phases.append(phase)

        if not return_trace:
            return h
        return h, {
            "states": torch.stack(states, dim=1),
            "gates": torch.stack(gates_trace, dim=1),
            "phases": torch.stack(phases, dim=0),
        }


class ReceptorCompositionModel(nn.Module):
    """Gate 0X2 model: object and quality meet only inside the BlockNeuron."""

    def __init__(self, config: ReceptorCompositionConfig) -> None:
        super().__init__()
        self.config = config
        self.object_embedding = nn.Embedding(config.num_classes, config.object_dim)
        self.attribute_embedding = nn.Embedding(
            config.num_attributes, config.attribute_dim
        )
        self.image_encoder = TinyImageEncoder(config.visual_dim)
        self.block = FactorSeparatedReceptorBlock(
            config.object_dim,
            config.attribute_dim,
            config.visual_dim,
            config.state_dim,
            config.branches,
            config.phase_dim,
            config.steps,
        )
        self.public_head = nn.Sequential(
            nn.Linear(config.state_dim, config.public_dim),
            nn.LayerNorm(config.public_dim),
        )
        self.detail_head = nn.Linear(config.state_dim, config.detail_dim)
        self.classifier = nn.Linear(config.public_dim, config.num_classes)
        self.attribute_classifier = nn.Linear(
            config.public_dim, config.num_attributes
        )
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
        if object_ids is None and attribute_ids is None and image is None:
            raise ValueError("at least one receptor or image modality is required")

        reference = object_ids
        if reference is None:
            reference = attribute_ids
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
            object_drive = torch.zeros(
                batch, self.config.object_dim, device=device
            )
            object_present = torch.zeros(batch, 1, device=device)

        if attribute_ids is not None:
            if attribute_ids.shape[0] != batch:
                raise ValueError("attribute_ids batch mismatch")
            attribute_drive = self.attribute_embedding(attribute_ids)
            attribute_present = torch.ones(batch, 1, device=device)
        else:
            attribute_drive = torch.zeros(
                batch, self.config.attribute_dim, device=device
            )
            attribute_present = torch.zeros(batch, 1, device=device)

        if image is not None:
            if image.shape[0] != batch:
                raise ValueError("image batch mismatch")
            visual_drive = self.image_encoder(image)
            image_present = torch.ones(batch, 1, device=device)
        else:
            visual_drive = torch.zeros(
                batch, self.config.visual_dim, device=device
            )
            image_present = torch.zeros(batch, 1, device=device)

        mask = torch.cat(
            [object_present, attribute_present, image_present], dim=-1
        )
        block_out = self.block(
            object_drive,
            attribute_drive,
            visual_drive,
            mask,
            phase_mode=phase_mode,
            return_trace=return_trace,
        )
        if return_trace:
            h, trace = block_out
        else:
            h = block_out
            trace = None

        public = F.normalize(self.public_head(h), dim=-1, eps=1e-6)
        detail = self.detail_head(h)
        result: dict[str, Tensor | dict[str, Tensor]] = {
            "state": h,
            "public": public,
            "detail": detail,
            "logits": self.classifier(public),
            "attr_logits": self.attribute_classifier(public),
        }
        if trace is not None:
            result["trace"] = trace
        return result

    def decode(self, public: Tensor, detail: Tensor | None = None) -> Tensor:
        if detail is None:
            detail = torch.zeros(
                public.shape[0], self.config.detail_dim, device=public.device
            )
        return self.decoder(torch.cat([public, detail], dim=-1))

    def decode_state(self, state: Tensor, *, keep_detail: bool = True) -> Tensor:
        public = F.normalize(self.public_head(state), dim=-1, eps=1e-6)
        detail = self.detail_head(state) if keep_detail else None
        return self.decode(public, detail)
