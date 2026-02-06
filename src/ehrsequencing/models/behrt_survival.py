"""
BEHRTForSurvival: BEHRT-based model for discrete-time survival analysis.

This module implements a survival analysis model that leverages pre-trained BEHRT
representations for predicting time-to-event outcomes from EHR sequences.

Key features:
- Uses pre-trained BEHRT encoder for rich patient representations
- Visit-level aggregation from code-level embeddings
- Support for frozen, LoRA, and full fine-tuning
- Outputs hazard at each visit for discrete-time survival
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from .behrt import BEHRT, BEHRTConfig


@dataclass
class BEHRTSurvivalConfig:
    """Configuration for BEHRTForSurvival model."""
    
    behrt_config: BEHRTConfig
    hazard_hidden_dim: int = 256
    dropout: float = 0.1
    freeze_behrt: bool = False
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    
    @classmethod
    def from_pretrained_small(cls, vocab_size: int, freeze_behrt: bool = False) -> 'BEHRTSurvivalConfig':
        """Small model config for local development."""
        return cls(
            behrt_config=BEHRTConfig.small(vocab_size),
            hazard_hidden_dim=128,
            dropout=0.1,
            freeze_behrt=freeze_behrt
        )
    
    @classmethod
    def from_pretrained_medium(cls, vocab_size: int, freeze_behrt: bool = False) -> 'BEHRTSurvivalConfig':
        """Medium model config."""
        return cls(
            behrt_config=BEHRTConfig.medium(vocab_size),
            hazard_hidden_dim=256,
            dropout=0.1,
            freeze_behrt=freeze_behrt
        )
    
    @classmethod
    def from_pretrained_large(cls, vocab_size: int, freeze_behrt: bool = False) -> 'BEHRTSurvivalConfig':
        """Large model config for cloud GPU."""
        return cls(
            behrt_config=BEHRTConfig.large(vocab_size),
            hazard_hidden_dim=512,
            dropout=0.1,
            freeze_behrt=freeze_behrt
        )


class BEHRTForSurvival(nn.Module):
    """
    BEHRT-based model for discrete-time survival analysis.
    
    Architecture:
        1. BEHRT encoder (pre-trained) - processes flattened code sequence
        2. Visit aggregation - pool code embeddings within each visit
        3. Hazard prediction head - predict hazard at each visit
    
    The model takes flattened sequences (all codes in one sequence) with visit IDs
    to maintain visit boundaries, then aggregates code-level representations to
    visit-level before predicting hazards.
    
    Args:
        config: BEHRTSurvivalConfig object
        pretrained_behrt: Optional pre-trained BEHRT encoder
    
    Example:
        >>> # Load pre-trained BEHRT
        >>> from ehrsequencing.models.behrt import BEHRTForMLM
        >>> pretrained = BEHRTForMLM.from_pretrained('checkpoints/behrt_mlm/')
        >>> 
        >>> # Create survival model
        >>> config = BEHRTSurvivalConfig.from_pretrained_small(vocab_size=1000)
        >>> model = BEHRTForSurvival(config, pretrained_behrt=pretrained.behrt)
        >>> 
        >>> # Forward pass
        >>> hazards = model(codes, ages, visit_ids, segment_ids, attention_mask)
    """
    
    def __init__(
        self,
        config: BEHRTSurvivalConfig,
        pretrained_behrt: Optional[BEHRT] = None
    ):
        super().__init__()
        
        self.config = config
        
        # BEHRT encoder
        if pretrained_behrt is not None:
            self.behrt = pretrained_behrt
        else:
            self.behrt = BEHRT(config.behrt_config)
        
        # Freeze BEHRT if specified
        if config.freeze_behrt:
            for param in self.behrt.parameters():
                param.requires_grad = False
        
        # Apply LoRA if specified
        if config.use_lora:
            from .lora import apply_lora_to_behrt
            self.behrt = apply_lora_to_behrt(
                self.behrt,
                rank=config.lora_rank,
                alpha=config.lora_alpha
            )
        
        # Hazard prediction head
        behrt_hidden_dim = config.behrt_config.hidden_dim
        self.hazard_head = nn.Sequential(
            nn.Linear(behrt_hidden_dim, config.hazard_hidden_dim),
            nn.LayerNorm(config.hazard_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hazard_hidden_dim, 1),
            nn.Sigmoid()  # Ensure hazard in (0, 1)
        )
    
    def aggregate_visits(
        self,
        code_embeddings: torch.Tensor,
        visit_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate code-level embeddings to visit-level.
        
        Args:
            code_embeddings: (batch, seq_len, hidden_dim) - BEHRT output
            visit_ids: (batch, seq_len) - Visit ID for each code
            attention_mask: (batch, seq_len) - 1 for real codes, 0 for padding
        
        Returns:
            visit_embeddings: (batch, max_visits, hidden_dim)
            visit_mask: (batch, max_visits) - 1 for real visits, 0 for padding
        """
        batch_size, seq_len, hidden_dim = code_embeddings.shape
        device = code_embeddings.device
        
        # Find max visit ID in batch
        max_visit_id = visit_ids.max().item() + 1
        
        # Initialize visit embeddings
        visit_embeddings = torch.zeros(
            batch_size, max_visit_id, hidden_dim,
            device=device,
            dtype=code_embeddings.dtype
        )
        
        # Initialize visit mask
        visit_mask = torch.zeros(
            batch_size, max_visit_id,
            device=device,
            dtype=torch.bool
        )
        
        # Aggregate codes within each visit (mean pooling)
        for b in range(batch_size):
            for v in range(max_visit_id):
                # Find codes belonging to this visit
                mask = (visit_ids[b] == v) & (attention_mask[b].bool())
                
                if mask.any():
                    # Mean pool codes in this visit
                    visit_embeddings[b, v] = code_embeddings[b, mask].mean(dim=0)
                    visit_mask[b, v] = True
        
        return visit_embeddings, visit_mask.float()
    
    def forward(
        self,
        codes: torch.Tensor,
        ages: torch.Tensor,
        visit_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_visit_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for survival prediction.
        
        Args:
            codes: (batch, seq_len) - Flattened code sequence
            ages: (batch, seq_len) - Age at each code
            visit_ids: (batch, seq_len) - Visit ID for each code
            segment_ids: (batch, seq_len) - Segment ID (always 0 for single sequence)
            attention_mask: (batch, seq_len) - 1 for real codes, 0 for padding
            return_visit_embeddings: If True, return visit embeddings along with hazards
        
        Returns:
            hazards: (batch, max_visits) - Hazard at each visit
            OR
            (hazards, visit_embeddings) if return_visit_embeddings=True
        """
        # BEHRT encoding (bidirectional context within visits)
        code_embeddings = self.behrt(
            codes=codes,
            ages=ages,
            segments=segment_ids,
            attention_mask=attention_mask
        )  # (batch, seq_len, hidden_dim)
        
        # Aggregate to visit-level
        visit_embeddings, visit_mask = self.aggregate_visits(
            code_embeddings, visit_ids, attention_mask
        )  # (batch, max_visits, hidden_dim), (batch, max_visits)
        
        # Predict hazard at each visit
        hazards = self.hazard_head(visit_embeddings).squeeze(-1)
        # Shape: (batch, max_visits)
        
        # Mask padding visits
        hazards = hazards * visit_mask
        
        if return_visit_embeddings:
            return hazards, visit_embeddings
        
        return hazards
    
    def compute_risk_score(
        self,
        hazards: torch.Tensor,
        time_horizon: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute risk score from hazards for ranking/evaluation.
        
        Args:
            hazards: (batch, max_visits) - Predicted hazards
            time_horizon: Optional time point for risk (None = cumulative)
        
        Returns:
            risk_scores: (batch,) - Risk scores for ranking
        """
        if time_horizon is not None:
            # Risk at specific time horizon: 1 - S(t)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk_scores = 1 - survival[:, time_horizon]
        else:
            # Cumulative risk (sum of hazards)
            risk_scores = hazards.sum(dim=1)
        
        return risk_scores
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        config: BEHRTSurvivalConfig,
        strict: bool = True
    ) -> 'BEHRTForSurvival':
        """
        Load BEHRTForSurvival from checkpoint.
        
        Args:
            pretrained_path: Path to checkpoint
            config: BEHRTSurvivalConfig
            strict: Whether to strictly enforce state dict keys
        
        Returns:
            model: BEHRTForSurvival instance
        """
        model = cls(config)
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=strict)
        return model
    
    def save_pretrained(self, save_path: str):
        """Save model checkpoint."""
        torch.save(self.state_dict(), save_path)
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """
        Get count of trainable parameters by component.
        
        Returns:
            param_counts: Dictionary with parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        behrt_total = sum(p.numel() for p in self.behrt.parameters())
        behrt_trainable = sum(p.numel() for p in self.behrt.parameters() if p.requires_grad)
        
        head_total = sum(p.numel() for p in self.hazard_head.parameters())
        head_trainable = sum(p.numel() for p in self.hazard_head.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
            'behrt_total': behrt_total,
            'behrt_trainable': behrt_trainable,
            'behrt_frozen': behrt_total - behrt_trainable,
            'head_total': head_total,
            'head_trainable': head_trainable,
            'trainable_percentage': 100.0 * trainable / total if total > 0 else 0.0
        }
