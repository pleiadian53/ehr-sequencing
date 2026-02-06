"""
Loss functions for EHR sequence modeling.

This module provides loss functions for various prediction tasks:
- Discrete-time survival analysis
- Fixed-horizon progression prediction
- Multi-task learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiscreteTimeSurvivalLoss(nn.Module):
    """
    Discrete-time survival analysis loss function.
    
    This loss is appropriate for modeling disease progression, readmission,
    or other time-to-event outcomes in visit-based EHR sequences.
    
    The loss implements the negative log-likelihood for discrete-time survival:
    
    For each patient:
        L = sum_{t < T} log(1 - h_t) + [event_occurred] * log(h_T)
    
    Where:
        - h_t is the hazard (probability of event) at visit t
        - T is the time of event or censoring
        - event_occurred is 1 if event observed, 0 if censored
    
    References:
        - Singer & Willett (2003). Applied Longitudinal Data Analysis.
        - Tutz & Schmid (2016). Modeling Discrete Time-to-Event Data.
    
    Example:
        >>> loss_fn = DiscreteTimeSurvivalLoss()
        >>> hazards = model(visit_sequences)  # [batch_size, num_visits]
        >>> loss = loss_fn(hazards, event_times, event_indicators, sequence_mask)
    """
    
    def __init__(self, eps: float = 1e-7):
        """
        Args:
            eps: Small constant to avoid log(0). Default: 1e-7
        """
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        hazards: torch.Tensor,
        event_times: torch.Tensor,
        event_indicators: torch.Tensor,
        sequence_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute discrete-time survival loss.
        
        Args:
            hazards: Predicted hazards at each visit. Shape: [batch_size, max_visits]
                    Values should be in (0, 1), typically from sigmoid activation.
            event_times: Index of event or censoring for each patient. Shape: [batch_size]
                        0-indexed, e.g., event_times[i] = 5 means event at 6th visit.
            event_indicators: Whether event was observed (1) or censored (0). Shape: [batch_size]
            sequence_mask: Mask for valid visits (1) vs padding (0). Shape: [batch_size, max_visits]
        
        Returns:
            Scalar loss (negative log-likelihood, averaged over batch)
        
        Example:
            >>> hazards = torch.tensor([[0.1, 0.2, 0.3, 0.0],
            ...                         [0.15, 0.25, 0.0, 0.0]])
            >>> event_times = torch.tensor([2, 1])  # Events at visits 3 and 2
            >>> event_indicators = torch.tensor([1, 1])  # Both observed
            >>> mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
            >>> loss = loss_fn(hazards, event_times, event_indicators, mask)
        """
        batch_size, max_visits = hazards.shape
        
        # Clamp hazards to avoid log(0)
        hazards = torch.clamp(hazards, min=self.eps, max=1 - self.eps)
        
        # Create time index tensor [1, max_visits]
        time_idx = torch.arange(max_visits, device=hazards.device).unsqueeze(0)
        
        # Expand event times to [batch_size, 1]
        event_times_expanded = event_times.unsqueeze(1)
        
        # Mask for visits before event/censoring
        before_event_mask = (time_idx < event_times_expanded).float() * sequence_mask
        
        # Mask for event visit
        at_event_mask = (time_idx == event_times_expanded).float() * sequence_mask
        
        # Log-likelihood from survival (all visits before event)
        # Sum of log(1 - h_t) for all t < T
        survival_ll = torch.sum(
            torch.log(1 - hazards) * before_event_mask,
            dim=1
        )
        
        # Log-likelihood from event (only if event observed)
        # log(h_T) if event occurred at T
        event_ll = torch.sum(
            torch.log(hazards) * at_event_mask,
            dim=1
        ) * event_indicators.float()
        
        # Total log-likelihood per patient
        log_likelihood = survival_ll + event_ll
        
        # Return negative log-likelihood (to minimize)
        return -torch.mean(log_likelihood)


class FixedHorizonProgressionLoss(nn.Module):
    """
    Fixed-horizon progression prediction loss.
    
    This loss is for predicting whether an event occurs within a fixed time window
    (e.g., "will patient progress within next 6 months?").
    
    This is simpler than discrete-time survival but requires careful handling of
    censoring to avoid temporal leakage.
    
    Example:
        >>> loss_fn = FixedHorizonProgressionLoss(horizon_days=180)
        >>> predictions = model(visit_sequences)  # [batch_size, num_visits]
        >>> loss = loss_fn(predictions, labels, valid_mask)
    """
    
    def __init__(self, horizon_days: Optional[int] = None):
        """
        Args:
            horizon_days: Prediction horizon in days. Used for documentation only.
        """
        super().__init__()
        self.horizon_days = horizon_days
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute fixed-horizon progression loss.
        
        Args:
            logits: Predicted logits at each visit. Shape: [batch_size, max_visits]
            labels: Binary labels (1=event, 0=no event). Shape: [batch_size, max_visits]
                   Should be -1 or masked for censored visits.
            valid_mask: Mask for valid (non-censored) visits. Shape: [batch_size, max_visits]
        
        Returns:
            Scalar loss (averaged over valid visits)
        """
        # Compute BCE loss
        loss = self.bce_loss(logits, labels.float())
        
        # Mask out censored visits
        masked_loss = loss * valid_mask
        
        # Average over valid visits
        num_valid = valid_mask.sum().clamp(min=1)
        return masked_loss.sum() / num_valid


class MultiTaskSurvivalLoss(nn.Module):
    """
    Multi-task loss combining survival and auxiliary tasks.
    
    Example use cases:
        - Joint prediction of progression + readmission
        - Survival + next-visit prediction
        - Multiple competing risks
    
    Example:
        >>> loss_fn = MultiTaskSurvivalLoss(task_weights={'survival': 1.0, 'readmit': 0.5})
        >>> losses = {
        ...     'survival': survival_loss,
        ...     'readmit': readmit_loss
        ... }
        >>> total_loss = loss_fn(losses)
    """
    
    def __init__(self, task_weights: dict[str, float]):
        """
        Args:
            task_weights: Dictionary mapping task names to weights.
                         Example: {'survival': 1.0, 'readmission': 0.5}
        """
        super().__init__()
        self.task_weights = task_weights
    
    def forward(self, task_losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted sum of task losses.
        
        Args:
            task_losses: Dictionary mapping task names to loss tensors.
        
        Returns:
            Weighted sum of losses
        """
        total_loss = 0.0
        for task_name, loss in task_losses.items():
            if task_name in self.task_weights:
                total_loss += self.task_weights[task_name] * loss
        
        return total_loss


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss for survival analysis.
    
    Directly optimizes discrimination (C-index) by penalizing incorrectly
    ordered pairs of patients. Patient with earlier event should have higher risk.
    
    Loss = (1/|P|) * sum_{(i,j) in P} max(0, r_j - r_i + margin)
    
    where P is the set of comparable pairs (i has event before j).
    
    This loss directly optimizes ranking quality but may sacrifice calibration.
    Use hybrid loss (NLL + Ranking) for best of both worlds.
    
    Args:
        margin: Margin for ranking loss (default: 0.1)
    
    Example:
        >>> loss_fn = PairwiseRankingLoss(margin=0.1)
        >>> risk_scores = model.compute_risk_score(hazards)
        >>> loss = loss_fn(risk_scores, event_times, event_indicators)
    """
    
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        risk_scores: torch.Tensor,
        event_times: torch.Tensor,
        event_indicators: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise ranking loss (vectorized).
        
        Args:
            risk_scores: (batch_size,) - Predicted risk scores
            event_times: (batch_size,) - Event/censoring times
            event_indicators: (batch_size,) - 1 if event, 0 if censored
        
        Returns:
            loss: Scalar tensor
        """
        batch_size = len(risk_scores)
        device = risk_scores.device
        
        # Create pairwise comparison matrices
        # Shape: (batch_size, batch_size)
        risk_diff = risk_scores.unsqueeze(1) - risk_scores.unsqueeze(0)  # r_i - r_j
        time_diff = event_times.unsqueeze(1) - event_times.unsqueeze(0)  # t_i - t_j
        
        # Mask for comparable pairs
        # i has event and i occurs before or at same time as j
        event_mask = event_indicators.unsqueeze(1).float()  # (batch, 1)
        time_mask = (time_diff <= 0).float()  # i occurs before/at j
        comparable_mask = event_mask * time_mask
        
        # Remove diagonal (self-comparisons)
        comparable_mask = comparable_mask * (1 - torch.eye(batch_size, device=device))
        
        # Compute loss: penalize if r_j >= r_i (i.e., r_i - r_j <= 0)
        # We want r_i > r_j, so penalize max(0, -risk_diff + margin)
        pairwise_loss = torch.relu(-risk_diff + self.margin)
        
        # Apply mask and average
        masked_loss = pairwise_loss * comparable_mask
        n_pairs = comparable_mask.sum()
        
        if n_pairs > 0:
            return masked_loss.sum() / n_pairs
        else:
            return torch.tensor(0.0, device=device)


class HybridSurvivalLoss(nn.Module):
    """
    Hybrid loss: NLL + Pairwise Ranking.
    
    Combines calibration (NLL) and discrimination (ranking) for best of both worlds.
    
    Loss = lambda_nll * NLL + lambda_rank * Ranking
    
    Args:
        lambda_nll: Weight for NLL loss (default: 1.0)
        lambda_rank: Weight for ranking loss (default: 0.1)
        margin: Margin for ranking loss (default: 0.1)
        eps: Epsilon for NLL numerical stability (default: 1e-7)
    
    Example:
        >>> loss_fn = HybridSurvivalLoss(lambda_nll=1.0, lambda_rank=0.1)
        >>> hazards = model(codes, ages, visit_ids, segment_ids, attention_mask)
        >>> risk_scores = model.compute_risk_score(hazards)
        >>> loss, loss_dict = loss_fn(
        ...     hazards, risk_scores, event_times, event_indicators, sequence_mask
        ... )
    """
    
    def __init__(
        self,
        lambda_nll: float = 1.0,
        lambda_rank: float = 0.1,
        margin: float = 0.1,
        eps: float = 1e-7
    ):
        super().__init__()
        self.lambda_nll = lambda_nll
        self.lambda_rank = lambda_rank
        
        self.nll_loss = DiscreteTimeSurvivalLoss(eps=eps)
        self.rank_loss = PairwiseRankingLoss(margin=margin)
    
    def forward(
        self,
        hazards: torch.Tensor,
        risk_scores: torch.Tensor,
        event_times: torch.Tensor,
        event_indicators: torch.Tensor,
        sequence_mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute hybrid loss.
        
        Args:
            hazards: (batch, max_visits) - Predicted hazards for NLL
            risk_scores: (batch,) - Predicted risk scores for ranking
            event_times: (batch,) - Event/censoring times
            event_indicators: (batch,) - 1 if event, 0 if censored
            sequence_mask: (batch, max_visits) - Mask for valid visits
        
        Returns:
            total_loss: Scalar tensor
            loss_dict: Dictionary with individual loss components
        """
        # Compute individual losses
        nll = self.nll_loss(hazards, event_times, event_indicators, sequence_mask)
        rank = self.rank_loss(risk_scores, event_times, event_indicators)
        
        # Weighted combination
        total_loss = self.lambda_nll * nll + self.lambda_rank * rank
        
        # Return total loss and components for logging
        loss_dict = {
            'total': total_loss.item(),
            'nll': nll.item(),
            'rank': rank.item()
        }
        
        return total_loss, loss_dict


def concordance_index(
    hazards: torch.Tensor,
    event_times: torch.Tensor,
    event_indicators: torch.Tensor,
    sequence_mask: torch.Tensor = None,
) -> float:
    """
    Compute concordance index (C-index) for survival predictions.
    
    The C-index measures the fraction of pairs where the model correctly
    ranks patients by risk. Higher risk should predict earlier events.
    
    Args:
        hazards: Predicted hazards. Shape: [batch_size, max_visits]
        event_times: Event or censoring times. Shape: [batch_size]
        event_indicators: 1 if event observed, 0 if censored. Shape: [batch_size]
        sequence_mask: Optional mask for valid visits. Shape: [batch_size, max_visits]
    
    Returns:
        C-index value in [0, 1]. Higher is better. 0.5 = random.
    
    Interpretation:
        - Patient with earlier event should have higher cumulative risk
        - C-index = P(risk_i > risk_j | time_i < time_j, both events)
    
    Note:
        This is a simplified implementation. For production, use lifelines or scikit-survival.
    """
    batch_size = hazards.shape[0]
    
    # Compute cumulative risk scores (sum of hazards over all visits)
    if sequence_mask is not None:
        # Only sum over valid visits
        risk_scores = (hazards * sequence_mask.float()).sum(dim=1)
    else:
        risk_scores = hazards.sum(dim=1)
    
    # Count concordant and discordant pairs
    concordant = 0
    discordant = 0
    total = 0
    
    for i in range(batch_size):
        # Only use patients with observed events as index cases
        if event_indicators[i] == 0:
            continue
        
        for j in range(batch_size):
            if i == j:
                continue
            
            # Compare pairs where times are different
            if event_times[j] > event_times[i]:
                total += 1
                # Concordant: patient with earlier event has higher risk
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] < risk_scores[j]:
                    discordant += 1
                # Ties are ignored
    
    if total == 0:
        return 0.5  # No comparable pairs
    
    return concordant / total
