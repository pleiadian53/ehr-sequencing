"""
HierarchicalBEHRTForSurvival: Two-level encoder for discrete-time survival analysis.

Architecture (see docs/methods/discrete_time_survival_analysis/09_hierarchical_architecture.md):

    Stage A — Within-visit encoder (codes → visit embedding)
        CodeEmbedding:          (B, V, C) → (B, V, C, d)
        AttnPoolingVisitEncoder: (B, V, C, d) → (B, V, d)

    Stage B — Across-visit encoder (visit embeddings → patient timeline)
        VisitTimeEmbedding:     adds positional + age + Δt signals → (B, V, d)
        TimelineEncoder:        Transformer over visits → (B, V, d)

    Survival head:
        MLP + sigmoid → hazard per visit → (B, V)

Key design choices:
    - Attention pooling (Option B) for within-visit aggregation: learned importance
      weights via a global query vector, no cross-code attention needed.
    - NaN guard in AttnPoolingVisitEncoder: all-padding visits produce uniform
      zero output rather than NaN from softmax([-inf, ...]).
    - Δt embedding at visit level: captures irregular visit spacing explicitly.
    - Loss and evaluation reuse existing DiscreteTimeSurvivalLoss / concordance_index.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, field

from .embeddings import AgeEmbedding, TimeEmbedding


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HierarchicalSurvivalConfig:
    """Configuration for HierarchicalBEHRTForSurvival."""

    vocab_size: int
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers_visit: int = 2       # Transformer layers in TimelineEncoder
    max_visits: int = 50
    max_codes_per_visit: int = 30
    max_age: int = 100
    age_bin_size: int = 5
    max_time_delta: int = 365 * 5  # days
    time_bin_size: int = 30        # ~monthly bins
    hazard_hidden_dim: int = 64
    dropout: float = 0.1
    pad_token: int = 0

    @classmethod
    def small(cls, vocab_size: int) -> 'HierarchicalSurvivalConfig':
        return cls(vocab_size=vocab_size, hidden_dim=64, n_heads=2, n_layers_visit=1,
                   hazard_hidden_dim=32)

    @classmethod
    def medium(cls, vocab_size: int) -> 'HierarchicalSurvivalConfig':
        return cls(vocab_size=vocab_size, hidden_dim=128, n_heads=4, n_layers_visit=2,
                   hazard_hidden_dim=64)

    @classmethod
    def large(cls, vocab_size: int) -> 'HierarchicalSurvivalConfig':
        return cls(vocab_size=vocab_size, hidden_dim=256, n_heads=8, n_layers_visit=4,
                   hazard_hidden_dim=128)


# ---------------------------------------------------------------------------
# Stage A: Within-visit encoder
# ---------------------------------------------------------------------------

class CodeEmbedding(nn.Module):
    """
    Embed each code token within a visit.

    Combines:
        - code identity embedding (learned lookup)
        - intra-visit positional embedding (position of code within the visit)

    Args:
        vocab_size: Number of distinct medical codes.
        hidden_dim: Embedding dimension.
        max_codes_per_visit: Maximum codes per visit (C dimension).
        pad_token: Padding token ID (excluded from embedding gradient).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        max_codes_per_visit: int = 30,
        pad_token: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.code_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token)
        self.pos_embedding = nn.Embedding(max_codes_per_visit, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            codes: (B, V, C) — code IDs

        Returns:
            embeddings: (B, V, C, d)
        """
        B, V, C = codes.shape
        device = codes.device

        code_emb = self.code_embedding(codes)                          # (B, V, C, d)
        positions = torch.arange(C, device=device)                     # (C,)
        pos_emb = self.pos_embedding(positions)                        # (C, d)
        pos_emb = pos_emb.view(1, 1, C, -1).expand(B, V, -1, -1)     # (B, V, C, d)

        emb = self.layer_norm(code_emb + pos_emb)
        return self.dropout(emb)


class AttnPoolingVisitEncoder(nn.Module):
    """
    Aggregate code embeddings within a visit using learned attention pooling.

    A single global query vector q scores each code independently. The result
    is a weighted average of code embeddings, where weights are normalized
    within each visit via softmax.

    NaN guard: if an entire visit slot is padding (code_mask all False), the
    softmax would receive all -inf and produce NaN. We detect this case and
    replace those visit outputs with zero vectors.

    Args:
        hidden_dim: Embedding dimension d.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Parameter(torch.randn(hidden_dim))

    def forward(
        self,
        e: torch.Tensor,
        code_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            e:         (B, V, C, d) — code embeddings
            code_mask: (B, V, C)    — True for real codes, False for padding

        Returns:
            visit_emb: (B, V, d) — one vector per visit
        """
        scores = (self.q * torch.tanh(self.W(e))).sum(-1)             # (B, V, C)
        scores = scores.masked_fill(~code_mask, float('-inf'))

        # NaN guard: visits where all codes are padding → replace -inf with 0
        # so softmax produces uniform weights, then zero out via mask below.
        all_pad = ~code_mask.any(dim=-1, keepdim=True)                 # (B, V, 1)
        scores = scores.masked_fill(all_pad.expand_as(scores), 0.0)

        alpha = torch.softmax(scores, dim=-1)                          # (B, V, C)
        alpha = alpha * code_mask.float()                              # zero out pad visits

        return (alpha.unsqueeze(-1) * e).sum(dim=2)                    # (B, V, d)


# ---------------------------------------------------------------------------
# Stage B: Across-visit encoder
# ---------------------------------------------------------------------------

class VisitTimeEmbedding(nn.Module):
    """
    Add temporal signals to visit-level representations before the timeline encoder.

    Combines:
        - visit index positional embedding
        - age embedding (binned, one per visit)
        - time-delta embedding (days since previous visit, binned)

    Args:
        hidden_dim: Embedding dimension.
        max_visits: Maximum number of visits (V dimension).
        max_age: Maximum patient age in years.
        age_bin_size: Age bin width in years.
        max_time_delta: Maximum time delta in days.
        time_bin_size: Time bin width in days.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_visits: int = 50,
        max_age: int = 100,
        age_bin_size: int = 5,
        max_time_delta: int = 365 * 5,
        time_bin_size: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_visits, hidden_dim)
        self.age_embedding = AgeEmbedding(hidden_dim, max_age=max_age, age_bin_size=age_bin_size)
        self.delta_embedding = TimeEmbedding(hidden_dim, max_time_delta=max_time_delta,
                                             time_bin_size=time_bin_size)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        u: torch.Tensor,
        ages: torch.Tensor,
        time_deltas: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            u:           (B, V, d) — visit embeddings from Stage A
            ages:        (B, V)    — patient age at each visit
            time_deltas: (B, V)    — days since previous visit (0 for first visit)

        Returns:
            x: (B, V, d) — visit embeddings with temporal signals added
        """
        B, V, _ = u.shape
        device = u.device

        positions = torch.arange(V, device=device).unsqueeze(0)       # (1, V)
        pos_emb = self.pos_embedding(positions)                        # (1, V, d)

        age_emb = self.age_embedding(ages)                             # (B, V, d)
        delta_emb = self.delta_embedding(time_deltas.long())           # (B, V, d)

        x = self.layer_norm(u + pos_emb + age_emb + delta_emb)
        return self.dropout(x)


class TimelineEncoder(nn.Module):
    """
    Transformer encoder over visit-level representations.

    Treats each visit embedding as a token and applies multi-head self-attention
    across the patient's visit timeline. The visit_mask prevents padded visit
    slots from contributing to attention.

    Args:
        hidden_dim: Embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer encoder layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self,
        x: torch.Tensor,
        visit_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:          (B, V, d) — visit embeddings with temporal signals
            visit_mask: (B, V)    — True for real visits, False for padding

        Returns:
            z: (B, V, d) — contextualized visit representations
        """
        # TransformerEncoder expects src_key_padding_mask=True where tokens should be IGNORED
        padding_mask = ~visit_mask                                     # (B, V)
        return self.transformer(x, src_key_padding_mask=padding_mask)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class HierarchicalBEHRTForSurvival(nn.Module):
    """
    Two-level hierarchical model for discrete-time survival analysis on EHR data.

    Stage A: Within-visit encoder
        CodeEmbedding + AttnPoolingVisitEncoder → (B, V, d)

    Stage B: Across-visit encoder
        VisitTimeEmbedding + TimelineEncoder → (B, V, d)

    Survival head:
        MLP + sigmoid → hazard per visit → (B, V)

    The loss and evaluation interface is identical to BEHRTForSurvival:
    use DiscreteTimeSurvivalLoss with sequence_mask=visit_mask.

    Args:
        config: HierarchicalSurvivalConfig

    Example:
        >>> config = HierarchicalSurvivalConfig.small(vocab_size=1000)
        >>> model = HierarchicalBEHRTForSurvival(config)
        >>> hazards = model(codes, ages, time_deltas, code_mask, visit_mask)
        >>> # hazards: (B, V) — discrete-time hazard at each visit
    """

    def __init__(self, config: HierarchicalSurvivalConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim

        self.code_embedding = CodeEmbedding(
            vocab_size=config.vocab_size,
            hidden_dim=d,
            max_codes_per_visit=config.max_codes_per_visit,
            pad_token=config.pad_token,
            dropout=config.dropout,
        )
        self.visit_encoder = AttnPoolingVisitEncoder(hidden_dim=d)

        self.visit_time_embedding = VisitTimeEmbedding(
            hidden_dim=d,
            max_visits=config.max_visits,
            max_age=config.max_age,
            age_bin_size=config.age_bin_size,
            max_time_delta=config.max_time_delta,
            time_bin_size=config.time_bin_size,
            dropout=config.dropout,
        )
        self.timeline_encoder = TimelineEncoder(
            hidden_dim=d,
            n_heads=config.n_heads,
            n_layers=config.n_layers_visit,
            dropout=config.dropout,
        )

        self.hazard_head = nn.Sequential(
            nn.Linear(d, config.hazard_hidden_dim),
            nn.LayerNorm(config.hazard_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hazard_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        codes: torch.Tensor,
        ages: torch.Tensor,
        time_deltas: torch.Tensor,
        code_mask: torch.Tensor,
        visit_mask: torch.Tensor,
        return_visit_embeddings: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for survival prediction.

        Args:
            codes:       (B, V, C) long   — code IDs; 0 = padding
            ages:        (B, V)    float  — patient age at each visit
            time_deltas: (B, V)    float  — days since previous visit
            code_mask:   (B, V, C) bool   — True for real codes
            visit_mask:  (B, V)    bool   — True for real visits
            return_visit_embeddings: If True, also return (B, V, d) visit states.

        Returns:
            hazards: (B, V) — discrete-time hazard at each visit (0 for padded visits)
            OR (hazards, visit_embeddings) if return_visit_embeddings=True
        """
        # Stage A: codes → visit embeddings
        e = self.code_embedding(codes)                                 # (B, V, C, d)
        u = self.visit_encoder(e, code_mask)                          # (B, V, d)

        # Stage B: visit embeddings → contextualized visit states
        x = self.visit_time_embedding(u, ages, time_deltas)           # (B, V, d)
        z = self.timeline_encoder(x, visit_mask)                      # (B, V, d)

        # Survival head
        hazards = self.hazard_head(z).squeeze(-1)                     # (B, V)
        hazards = hazards * visit_mask.float()                        # zero padded visits

        if return_visit_embeddings:
            return hazards, z
        return hazards

    def compute_risk_score(
        self,
        hazards: torch.Tensor,
        time_horizon: Optional[int] = None,
        visit_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute risk score from hazards for ranking/evaluation.

        Args:
            hazards:      (B, V) — predicted hazards
            time_horizon: Optional visit index for fixed-horizon risk.
                          None = mean hazard over real visits (length-normalized).
            visit_mask:   (B, V) bool/float — real visits; used for normalization.
                          If None, all positions treated as real.

        Returns:
            risk_scores: (B,)
        """
        if time_horizon is not None:
            survival = torch.cumprod(1 - hazards, dim=1)
            return 1 - survival[:, time_horizon]

        if visit_mask is not None:
            mask = visit_mask.float()
            n_real = mask.sum(dim=1).clamp(min=1.0)
            return (hazards * mask).sum(dim=1) / n_real
        # Fallback: mean over all positions
        return hazards.mean(dim=1)

    def get_trainable_parameters(self) -> Dict[str, int]:
        """Parameter counts by component."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
            'trainable_percentage': 100.0 * trainable / total if total > 0 else 0.0,
        }

    def save_pretrained(self, save_path: str):
        """Save model checkpoint."""
        torch.save(self.state_dict(), save_path)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        config: 'HierarchicalSurvivalConfig',
        strict: bool = True,
    ) -> 'HierarchicalBEHRTForSurvival':
        """Load from checkpoint."""
        model = cls(config)
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=strict)
        return model
