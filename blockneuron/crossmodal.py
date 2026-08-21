from __future__ import annotations

"""Small cross-modal BlockNeuron components for Gate 0X.

This module deliberately avoids pretrained CLIP/text/image models. Text and image
receptors are learned jointly, enter the same recurrent block, and are forced to
agree only through a shared public state. The image decoder never receives text
features directly.
"""

from dataclasses import asdict, dataclass
import math
import string
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


DEFAULT_ALPHABET = " " + string.ascii_lowercase + string.digits + "-_/.,'"


@dataclass
class CrossModalConfig:
    image_size: int = 28
    num_classes: int = 10
    text_embed_dim: int = 16
    semantic_dim: int = 32
    visual_dim: int = 48
    state_dim: int = 64
    public_dim: int = 24
    detail_dim: int = 32
    branches: int = 8
    phase_dim: int = 4
    steps: int = 8
    basis_dim: int = 128
    max_text_len: int = 40

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class CharTokenizer:
    """Tiny deterministic character tokenizer so queries are actual strings."""

    def __init__(self, alphabet: str = DEFAULT_ALPHABET, max_len: int = 40) -> None:
        self.alphabet = alphabet
        self.max_len = int(max_len)
        self.stoi = {ch: i + 1 for i, ch in enumerate(alphabet)}

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet) + 1

    def encode(self, texts: Sequence[str], *, device: torch.device | str | None = None) -> Tensor:
        out = torch.zeros(len(texts), self.max_len, dtype=torch.long, device=device)
        for row, text in enumerate(texts):
            clean = text.lower().strip()[: self.max_len]
            if not clean:
                continue
            ids = [self.stoi.get(ch, 0) for ch in clean]
            out[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        return out


class TinyTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, semantic_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, semantic_dim, batch_first=True)
        self.norm = nn.LayerNorm(semantic_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        emb = self.embedding(tokens)
        seq, _ = self.gru(emb)
        lengths = (tokens != 0).sum(dim=1).clamp_min(1)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, seq.shape[-1])
        last = seq.gather(1, idx).squeeze(1)
        return self.norm(last)


class TinyImageEncoder(nn.Module):
    def __init__(self, visual_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(32 * 4 * 4, visual_dim), nn.LayerNorm(visual_dim))

    def forward(self, image: Tensor) -> Tensor:
        return self.proj(self.conv(image))


class ResonantCrossModalBlock(nn.Module):
    """One shared recurrent block with modality receptors and passing phase modes.

    Text and image drives land on the same branch bank. Branch selection depends on
    content-dependent receptor match plus a global multi-phase trajectory. There is
    no concept->phase lookup and no text->decoder skip connection.
    """

    def __init__(
        self,
        semantic_dim: int,
        visual_dim: int,
        state_dim: int,
        branches: int,
        phase_dim: int,
        steps: int,
    ) -> None:
        super().__init__()
        self.semantic_dim = semantic_dim
        self.visual_dim = visual_dim
        self.state_dim = state_dim
        self.branches = branches
        self.phase_dim = phase_dim
        self.steps = steps

        drive_dim = semantic_dim + visual_dim + 2
        self.semantic_receptors = nn.Parameter(torch.randn(branches, semantic_dim) * 0.25)
        self.visual_receptors = nn.Parameter(torch.randn(branches, visual_dim) * 0.25)
        self.mask_receptors = nn.Parameter(torch.randn(branches, 2) * 0.1)

        self.input_weight = nn.Parameter(torch.empty(branches, state_dim, drive_dim))
        self.recurrent_weight = nn.Parameter(torch.empty(branches, state_dim, state_dim))
        self.branch_bias = nn.Parameter(torch.zeros(branches, state_dim))
        self.cable = nn.Parameter(torch.ones(branches))

        self.phase_preference = nn.Parameter(torch.empty(branches, phase_dim))
        self.phase0 = nn.Parameter(torch.zeros(phase_dim))
        # Incommensurate-ish initial rates: fixed trajectory initially, trainable thereafter.
        base_omega = torch.tensor([0.53, 0.79, 1.11, 1.41])
        if phase_dim <= len(base_omega):
            omega = base_omega[:phase_dim].clone()
        else:
            omega = torch.cat([base_omega, torch.linspace(1.57, 2.17, phase_dim - 4)], dim=0)
        self.omega = nn.Parameter(omega)

        self.update_logit = nn.Parameter(torch.tensor(-0.45))
        self.state_norm = nn.LayerNorm(state_dim)

        nn.init.xavier_uniform_(self.input_weight)
        for branch in range(branches):
            nn.init.orthogonal_(self.recurrent_weight[branch])
        nn.init.uniform_(self.phase_preference, -math.pi, math.pi)

    def _content_score(self, semantic: Tensor, visual: Tensor, mask: Tensor) -> Tensor:
        sem = F.normalize(semantic, dim=-1, eps=1e-6)
        vis = F.normalize(visual, dim=-1, eps=1e-6)
        sem_r = F.normalize(self.semantic_receptors, dim=-1, eps=1e-6)
        vis_r = F.normalize(self.visual_receptors, dim=-1, eps=1e-6)
        sem_score = sem @ sem_r.T
        vis_score = vis @ vis_r.T
        return sem_score + vis_score + mask @ self.mask_receptors.T

    def forward(
        self,
        semantic: Tensor,
        visual: Tensor,
        mask: Tensor,
        *,
        phase_mode: str = "dynamic",
        return_trace: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if phase_mode not in {"dynamic", "clamped"}:
            raise ValueError("phase_mode must be 'dynamic' or 'clamped'")
        if semantic.shape[0] != visual.shape[0] or semantic.shape[0] != mask.shape[0]:
            raise ValueError("semantic, visual and mask batch dimensions must match")

        batch = semantic.shape[0]
        h = semantic.new_zeros(batch, self.state_dim)
        content_score = self._content_score(semantic, visual, mask)
        drive = torch.cat([semantic, visual, mask], dim=-1)
        input_drive = torch.einsum("nm,rdm->nrd", drive, self.input_weight) + self.branch_bias
        alpha = torch.sigmoid(self.update_logit)

        states: list[Tensor] = []
        gates_trace: list[Tensor] = []
        phases: list[Tensor] = []

        for step in range(self.steps):
            if phase_mode == "dynamic":
                phase = self.phase0 + float(step) * self.omega
            else:
                phase = torch.zeros_like(self.phase0)
            phase_score = torch.cos(phase[None, None, :] - self.phase_preference[None, :, :]).mean(-1)
            gates = torch.softmax(2.0 * content_score + 2.0 * phase_score, dim=-1)

            recurrent = torch.einsum("ni,rdi->nrd", h, self.recurrent_weight)
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
        trace = {
            "states": torch.stack(states, dim=1),
            "gates": torch.stack(gates_trace, dim=1),
            "phases": torch.stack(phases, dim=0),
        }
        return h, trace


class CoordinateImageDecoder(nn.Module):
    """Low-rank coordinate decoder: shared basis + state-generated coefficients."""

    def __init__(self, image_size: int, latent_dim: int, basis_dim: int) -> None:
        super().__init__()
        self.image_size = image_size
        self.basis_dim = basis_dim

        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, image_size),
            torch.linspace(-1.0, 1.0, image_size),
            indexing="ij",
        )
        coords = torch.stack([xs, ys], dim=-1).reshape(-1, 2)
        self.register_buffer("coords", coords, persistent=False)

        self.basis = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, basis_dim),
            nn.Tanh(),
        )
        self.coeff = nn.Linear(latent_dim, basis_dim)
        self.bias = nn.Linear(latent_dim, 1)

    def forward(self, latent: Tensor) -> Tensor:
        basis = self.basis(self.coords)
        coeff = self.coeff(latent)
        logits = coeff @ basis.T / math.sqrt(self.basis_dim)
        logits = logits + self.bias(latent)
        image = torch.sigmoid(logits)
        return image.view(latent.shape[0], 1, self.image_size, self.image_size)


class CrossModalBlockModel(nn.Module):
    def __init__(self, config: CrossModalConfig, *, alphabet: str = DEFAULT_ALPHABET) -> None:
        super().__init__()
        self.config = config
        self.alphabet = alphabet
        self.text_encoder = TinyTextEncoder(
            len(alphabet) + 1, config.text_embed_dim, config.semantic_dim
        )
        self.image_encoder = TinyImageEncoder(config.visual_dim)
        self.block = ResonantCrossModalBlock(
            config.semantic_dim,
            config.visual_dim,
            config.state_dim,
            config.branches,
            config.phase_dim,
            config.steps,
        )
        self.public_head = nn.Sequential(
            nn.Linear(config.state_dim, config.public_dim), nn.LayerNorm(config.public_dim)
        )
        self.detail_head = nn.Linear(config.state_dim, config.detail_dim)
        self.classifier = nn.Linear(config.public_dim, config.num_classes)
        self.decoder = CoordinateImageDecoder(
            config.image_size,
            config.public_dim + config.detail_dim,
            config.basis_dim,
        )

    def encode(
        self,
        *,
        tokens: Tensor | None = None,
        image: Tensor | None = None,
        phase_mode: str = "dynamic",
        return_trace: bool = False,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        if tokens is None and image is None:
            raise ValueError("at least one modality is required")
        if tokens is not None:
            batch = tokens.shape[0]
            device = tokens.device
            semantic = self.text_encoder(tokens)
            text_present = torch.ones(batch, 1, device=device)
        else:
            assert image is not None
            batch = image.shape[0]
            device = image.device
            semantic = torch.zeros(batch, self.config.semantic_dim, device=device)
            text_present = torch.zeros(batch, 1, device=device)

        if image is not None:
            visual = self.image_encoder(image)
            image_present = torch.ones(batch, 1, device=device)
        else:
            visual = torch.zeros(batch, self.config.visual_dim, device=device)
            image_present = torch.zeros(batch, 1, device=device)

        mask = torch.cat([text_present, image_present], dim=-1)
        block_out = self.block(
            semantic,
            visual,
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
        }
        if trace is not None:
            result["trace"] = trace
        return result

    def decode(self, public: Tensor, detail: Tensor | None = None) -> Tensor:
        if detail is None:
            detail = torch.zeros(public.shape[0], self.config.detail_dim, device=public.device)
        return self.decoder(torch.cat([public, detail], dim=-1))

    def decode_state(self, state: Tensor, *, keep_detail: bool = True) -> Tensor:
        public = F.normalize(self.public_head(state), dim=-1, eps=1e-6)
        detail = self.detail_head(state) if keep_detail else None
        return self.decode(public, detail)
