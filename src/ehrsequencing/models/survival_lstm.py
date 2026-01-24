"""
LSTM model for discrete-time survival analysis on EHR sequences.

This module implements an LSTM-based model for predicting disease progression,
readmission, or other time-to-event outcomes from visit-based EHR sequences.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class DiscreteTimeSurvivalLSTM(nn.Module):
    """
    LSTM model for discrete-time survival analysis.
    
    This model processes visit-based EHR sequences and outputs a hazard
    (probability of event) at each visit. The hazard represents:
    
        h_t = P(T = t | T >= t) = P(event at time t | survived to t)
    
    For forward prediction from time t, the hazard at the next time point
    h_{t+1} represents the future risk (using data through time t+1).
    
    Architecture:
        1. Embed medical codes (diagnoses, procedures, medications)
        2. Aggregate codes within each visit (mean pooling)
        3. LSTM over visit sequence
        4. Map LSTM hidden states to hazards (sigmoid activation)
    
    Example:
        >>> model = DiscreteTimeSurvivalLSTM(
        ...     vocab_size=5000,
        ...     embedding_dim=128,
        ...     hidden_dim=256,
        ...     num_layers=2
        ... )
        >>> hazards = model(visit_codes, visit_mask, sequence_mask)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        """
        Args:
            vocab_size: Size of medical code vocabulary
            embedding_dim: Dimension of code embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability (applied between LSTM layers)
            bidirectional: Whether to use bidirectional LSTM
                          Note: For causal modeling, this should be False
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Code embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # LSTM over visit sequence
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Hazard prediction head
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.hazard_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Ensure hazard in (0, 1)
        )
    
    def forward(
        self,
        visit_codes: torch.Tensor,
        visit_mask: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            visit_codes: Medical codes for each visit. 
                        Shape: [batch_size, num_visits, max_codes_per_visit]
            visit_mask: Mask for valid codes within each visit (1=valid, 0=padding).
                       Shape: [batch_size, num_visits, max_codes_per_visit]
            sequence_mask: Mask for valid visits (1=valid, 0=padding).
                          Shape: [batch_size, num_visits]
        
        Returns:
            hazards: Predicted hazard at each visit. Shape: [batch_size, num_visits]
        """
        batch_size, num_visits, max_codes = visit_codes.shape
        
        # Embed codes: [B, V, C, E]
        embeddings = self.embedding(visit_codes)
        
        # Aggregate codes within each visit (mean pooling)
        # Expand mask for broadcasting: [B, V, C, 1]
        visit_mask_expanded = visit_mask.unsqueeze(-1).float()
        
        # Sum embeddings and normalize by number of codes
        visit_vectors = (embeddings * visit_mask_expanded).sum(dim=2)  # [B, V, E]
        num_codes_per_visit = visit_mask.sum(dim=2, keepdim=True).clamp(min=1)  # [B, V, 1]
        visit_vectors = visit_vectors / num_codes_per_visit  # [B, V, E]
        
        # LSTM over visits: [B, V, H]
        lstm_out, _ = self.lstm(visit_vectors)
        
        # Map to hazards: [B, V, 1] -> [B, V]
        hazards = self.hazard_head(lstm_out).squeeze(-1)
        
        # Mask out padding visits
        hazards = hazards * sequence_mask.float()
        
        return hazards
    
    def predict_survival(
        self,
        visit_codes: torch.Tensor,
        visit_mask: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict survival probabilities and cumulative hazards.
        
        Args:
            visit_codes, visit_mask, sequence_mask: Same as forward()
        
        Returns:
            survival_probs: S_t = prod_{k=1}^t (1 - h_k). Shape: [batch_size, num_visits]
            cumulative_hazards: H_t = sum_{k=1}^t h_k. Shape: [batch_size, num_visits]
        """
        # Get hazards
        hazards = self.forward(visit_codes, visit_mask, sequence_mask)
        
        # Compute survival probabilities: S_t = prod_{k<=t} (1 - h_k)
        # Use cumsum of log(1 - h) for numerical stability
        log_survival = torch.cumsum(torch.log(1 - hazards + 1e-7), dim=1)
        survival_probs = torch.exp(log_survival)
        
        # Compute cumulative hazards: H_t = sum_{k<=t} h_k
        cumulative_hazards = torch.cumsum(hazards, dim=1)
        
        return survival_probs, cumulative_hazards


class DiscreteTimeSurvivalLSTMWithTime(DiscreteTimeSurvivalLSTM):
    """
    Extension of DiscreteTimeSurvivalLSTM that incorporates time information.
    
    This model includes:
        - Time since last visit
        - Time since first visit
        - Visit index
    
    Useful when visit spacing is irregular and time information is predictive.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        time_encoding_dim: int = 16,
    ):
        """
        Args:
            vocab_size: Size of medical code vocabulary
            embedding_dim: Dimension of code embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            time_encoding_dim: Dimension for time feature encoding
        """
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=False
        )
        
        # Time feature encoder
        self.time_encoder = nn.Sequential(
            nn.Linear(3, time_encoding_dim),  # 3 time features
            nn.ReLU(),
            nn.Linear(time_encoding_dim, time_encoding_dim)
        )
        
        # Update LSTM input size to include time features
        self.lstm = nn.LSTM(
            input_size=embedding_dim + time_encoding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
    
    def forward(
        self,
        visit_codes: torch.Tensor,
        visit_mask: torch.Tensor,
        sequence_mask: torch.Tensor,
        time_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with time features.
        
        Args:
            visit_codes: Medical codes. Shape: [batch_size, num_visits, max_codes_per_visit]
            visit_mask: Code mask. Shape: [batch_size, num_visits, max_codes_per_visit]
            sequence_mask: Visit mask. Shape: [batch_size, num_visits]
            time_features: Time features. Shape: [batch_size, num_visits, 3]
                          Features: [time_since_last_visit, time_since_first_visit, visit_index]
                          If None, will be set to zeros.
        
        Returns:
            hazards: Predicted hazard at each visit. Shape: [batch_size, num_visits]
        """
        batch_size, num_visits, max_codes = visit_codes.shape
        
        # Embed codes
        embeddings = self.embedding(visit_codes)
        
        # Aggregate codes within each visit
        visit_mask_expanded = visit_mask.unsqueeze(-1).float()
        visit_vectors = (embeddings * visit_mask_expanded).sum(dim=2)
        num_codes_per_visit = visit_mask.sum(dim=2, keepdim=True).clamp(min=1)
        visit_vectors = visit_vectors / num_codes_per_visit  # [B, V, E]
        
        # Encode time features
        if time_features is None:
            time_features = torch.zeros(batch_size, num_visits, 3, device=visit_codes.device)
        
        time_encoded = self.time_encoder(time_features)  # [B, V, T]
        
        # Concatenate visit vectors and time features
        lstm_input = torch.cat([visit_vectors, time_encoded], dim=-1)  # [B, V, E+T]
        
        # LSTM over visits
        lstm_out, _ = self.lstm(lstm_input)
        
        # Map to hazards
        hazards = self.hazard_head(lstm_out).squeeze(-1)
        
        # Mask out padding
        hazards = hazards * sequence_mask.float()
        
        return hazards


class CompetingRisksSurvivalLSTM(nn.Module):
    """
    LSTM model for competing risks survival analysis.
    
    Handles multiple event types (e.g., progression, death, readmission)
    where only one can occur first.
    
    Outputs cause-specific hazards for each event type.
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_event_types: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        """
        Args:
            vocab_size: Size of medical code vocabulary
            num_event_types: Number of competing event types
            embedding_dim: Dimension of code embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_event_types = num_event_types
        
        # Code embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Cause-specific hazard heads
        self.hazard_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            for _ in range(num_event_types)
        ])
    
    def forward(
        self,
        visit_codes: torch.Tensor,
        visit_mask: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            visit_codes, visit_mask, sequence_mask: Same as DiscreteTimeSurvivalLSTM
        
        Returns:
            hazards: Cause-specific hazards. Shape: [batch_size, num_visits, num_event_types]
        """
        batch_size, num_visits, max_codes = visit_codes.shape
        
        # Embed and aggregate
        embeddings = self.embedding(visit_codes)
        visit_mask_expanded = visit_mask.unsqueeze(-1).float()
        visit_vectors = (embeddings * visit_mask_expanded).sum(dim=2)
        num_codes_per_visit = visit_mask.sum(dim=2, keepdim=True).clamp(min=1)
        visit_vectors = visit_vectors / num_codes_per_visit
        
        # LSTM
        lstm_out, _ = self.lstm(visit_vectors)
        
        # Compute cause-specific hazards
        hazards_list = []
        for head in self.hazard_heads:
            h = head(lstm_out).squeeze(-1)  # [B, V]
            hazards_list.append(h)
        
        hazards = torch.stack(hazards_list, dim=-1)  # [B, V, K]
        
        # Mask padding
        sequence_mask_expanded = sequence_mask.unsqueeze(-1).float()
        hazards = hazards * sequence_mask_expanded
        
        return hazards
