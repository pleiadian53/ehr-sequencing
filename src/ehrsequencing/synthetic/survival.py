"""
Synthetic survival outcome generators.

This module provides classes for generating realistic synthetic survival
outcomes from patient sequences for training and evaluation.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch
from dataclasses import dataclass

from ..data.sequence_builder import PatientSequence


@dataclass
class SurvivalOutcome:
    """
    Container for survival outcome data.
    
    Attributes:
        event_times: Event or censoring times (visit indices)
        event_indicators: 1 if event observed, 0 if censored
        risk_scores: Optional computed risk scores for each patient
        metadata: Optional additional information about outcome generation
    """
    event_times: torch.Tensor
    event_indicators: torch.Tensor
    risk_scores: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None


class DiscreteTimeSurvivalGenerator:
    """
    Generate synthetic discrete-time survival outcomes.
    
    Creates realistic disease progression patterns where risk depends on
    patient characteristics extracted from their visit history:
    - Comorbidity burden (number of codes)
    - Visit frequency (healthcare utilization)
    - Code diversity (condition complexity)
    
    This generator is suitable for discrete-time survival models where
    hazard is predicted at each visit.
    
    Example:
        >>> generator = DiscreteTimeSurvivalGenerator(
        ...     censoring_rate=0.3,
        ...     risk_weights={'comorbidity': 0.4, 'frequency': 0.4, 'diversity': 0.2}
        ... )
        >>> outcome = generator.generate(sequences)
        >>> print(f"Event rate: {outcome.event_indicators.float().mean():.2%}")
    """
    
    def __init__(
        self,
        censoring_rate: float = 0.3,
        risk_weights: Optional[Dict[str, float]] = None,
        time_scale: float = 0.3,
        seed: Optional[int] = 42,
    ):
        """
        Initialize discrete-time survival generator.
        
        Args:
            censoring_rate: Proportion of patients to censor (0-1)
            risk_weights: Weights for risk factors. Keys: 'comorbidity', 'frequency', 'diversity'
                         Default: {'comorbidity': 0.4, 'frequency': 0.4, 'diversity': 0.2}
            time_scale: Scale factor for time-to-event (higher = events occur later)
            seed: Random seed for reproducibility
        """
        self.censoring_rate = censoring_rate
        self.risk_weights = risk_weights or {
            'comorbidity': 0.4,
            'frequency': 0.4,
            'diversity': 0.2
        }
        self.time_scale = time_scale
        self.seed = seed
        
        # Normalization constants (typical maximum values)
        self.norm_constants = {
            'comorbidity': 20.0,  # avg codes per visit
            'frequency': 5.0,     # visits per year
        }
    
    def generate(self, sequences: List[PatientSequence]) -> SurvivalOutcome:
        """
        Generate synthetic survival outcomes for patient sequences.
        
        Args:
            sequences: List of PatientSequence objects
        
        Returns:
            SurvivalOutcome with event times, indicators, and risk scores
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        
        event_times = []
        event_indicators = []
        risk_scores = []
        
        for seq in sequences:
            # Compute risk factors
            risk_factors = self._compute_risk_factors(seq)
            
            # Compute overall risk score
            risk_score = self._compute_risk_score(risk_factors)
            risk_scores.append(risk_score)
            
            # Simulate time-to-event
            event_time = self._simulate_event_time(risk_score, len(seq.visits))
            
            # Apply censoring
            is_censored = np.random.random() < self.censoring_rate
            
            if is_censored:
                # Censor at random time before potential event
                censor_time = np.random.randint(0, len(seq.visits))
                event_times.append(censor_time)
                event_indicators.append(0)
            else:
                # Event occurs
                event_times.append(event_time)
                event_indicators.append(1)
        
        return SurvivalOutcome(
            event_times=torch.tensor(event_times),
            event_indicators=torch.tensor(event_indicators),
            risk_scores=torch.tensor(risk_scores),
            metadata={
                'censoring_rate': self.censoring_rate,
                'risk_weights': self.risk_weights,
                'time_scale': self.time_scale,
                'seed': self.seed,
            }
        )
    
    def _compute_risk_factors(self, seq: PatientSequence) -> Dict[str, float]:
        """
        Extract risk factors from patient sequence.
        
        Args:
            seq: PatientSequence object
        
        Returns:
            Dictionary of risk factors
        """
        num_visits = len(seq.visits)
        
        # 1. Comorbidity burden (average codes per visit)
        avg_codes = np.mean([visit.num_codes() for visit in seq.visits])
        
        # 2. Visit frequency (visits per unit time)
        if num_visits > 1:
            time_span = (seq.visits[-1].timestamp - seq.visits[0].timestamp).days
            visit_frequency = num_visits / max(time_span / 365.0, 0.1)  # visits per year
        else:
            visit_frequency = 1.0
        
        # 3. Code diversity (unique codes / total codes)
        all_codes = []
        for visit in seq.visits:
            all_codes.extend(visit.get_all_codes())
        code_diversity = len(set(all_codes)) / max(len(all_codes), 1)
        
        return {
            'comorbidity': avg_codes,
            'frequency': visit_frequency,
            'diversity': code_diversity,
        }
    
    def _compute_risk_score(self, risk_factors: Dict[str, float]) -> float:
        """
        Compute overall risk score from risk factors.
        
        High risk: many codes, frequent visits, low diversity (repeated conditions)
        
        Args:
            risk_factors: Dictionary of risk factors
        
        Returns:
            Risk score in [0.1, 0.9]
        """
        risk_score = (
            self.risk_weights['comorbidity'] * (risk_factors['comorbidity'] / self.norm_constants['comorbidity']) +
            self.risk_weights['frequency'] * (risk_factors['frequency'] / self.norm_constants['frequency']) +
            self.risk_weights['diversity'] * (1 - risk_factors['diversity'])  # Low diversity = high risk
        )
        
        return np.clip(risk_score, 0.1, 0.9)
    
    def _simulate_event_time(self, risk_score: float, num_visits: int) -> int:
        """
        Simulate time-to-event with strong risk-time correlation.
        
        Higher risk_score = shorter time to event (negative correlation)
        
        Args:
            risk_score: Patient risk score in [0.1, 0.9]
            num_visits: Total number of visits
        
        Returns:
            Event time (visit index)
        """
        # Normalize risk_score from [0.1, 0.9] to [0, 1] for better spread
        normalized_risk = (risk_score - 0.1) / 0.8  # Now in [0, 1]
        
        # Base event time: inversely proportional to risk
        # High risk (1.0) → early event (visit ~0)
        # Low risk (0.0) → late event (visit ~max)
        base_time_fraction = 1.0 - normalized_risk
        
        # Add small controlled noise to preserve correlation
        noise_std = 0.08  # Small noise to preserve strong correlation
        noise = np.random.normal(0, noise_std)
        noisy_fraction = np.clip(base_time_fraction + noise, 0.02, 0.98)
        
        # Scale to full visit range (use more of the available visits)
        # This ensures high-risk and low-risk patients have clearly different event times
        max_event_visit = max(5, num_visits - 1)
        event_time = int(noisy_fraction * max_event_visit * self.time_scale * 2)
        
        # Ensure within valid range
        event_visit = int(np.clip(event_time, 0, num_visits - 1))
        
        return event_visit


class ContinuousTimeSurvivalGenerator:
    """
    Generate synthetic continuous-time survival outcomes.
    
    Suitable for Cox proportional hazards models and other continuous-time
    survival models. Events can occur at any time point, not just at visits.
    
    Example:
        >>> generator = ContinuousTimeSurvivalGenerator(baseline_hazard=0.01)
        >>> outcome = generator.generate(sequences)
    """
    
    def __init__(
        self,
        baseline_hazard: float = 0.01,
        censoring_rate: float = 0.3,
        seed: Optional[int] = 42,
    ):
        """
        Initialize continuous-time survival generator.
        
        Args:
            baseline_hazard: Baseline hazard rate (events per day)
            censoring_rate: Proportion of patients to censor
            seed: Random seed
        """
        self.baseline_hazard = baseline_hazard
        self.censoring_rate = censoring_rate
        self.seed = seed
    
    def generate(self, sequences: List[PatientSequence]) -> SurvivalOutcome:
        """
        Generate continuous-time survival outcomes.
        
        Args:
            sequences: List of PatientSequence objects
        
        Returns:
            SurvivalOutcome with event times in days (not visit indices)
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        
        event_times = []
        event_indicators = []
        
        for seq in sequences:
            # Compute risk factors (reuse discrete-time logic)
            generator = DiscreteTimeSurvivalGenerator(seed=None)
            risk_factors = generator._compute_risk_factors(seq)
            risk_score = generator._compute_risk_score(risk_factors)
            
            # Total follow-up time in days
            if len(seq.visits) > 1:
                follow_up_days = (seq.visits[-1].timestamp - seq.visits[0].timestamp).days
            else:
                follow_up_days = 365  # Default 1 year
            
            # Simulate event time using exponential distribution
            # Hazard = baseline_hazard * exp(risk_score)
            hazard = self.baseline_hazard * np.exp(risk_score)
            time_to_event_days = np.random.exponential(1.0 / hazard)
            
            # Apply censoring
            is_censored = np.random.random() < self.censoring_rate
            
            if is_censored or time_to_event_days > follow_up_days:
                # Censored at end of follow-up
                event_times.append(follow_up_days)
                event_indicators.append(0)
            else:
                # Event occurs
                event_times.append(time_to_event_days)
                event_indicators.append(1)
        
        return SurvivalOutcome(
            event_times=torch.tensor(event_times),
            event_indicators=torch.tensor(event_indicators),
            metadata={
                'baseline_hazard': self.baseline_hazard,
                'censoring_rate': self.censoring_rate,
                'time_unit': 'days',
            }
        )


class CompetingRisksGenerator:
    """
    Generate synthetic competing risks outcomes.
    
    Multiple event types can occur, and the occurrence of one event
    precludes the others (e.g., death from different causes).
    
    Example:
        >>> generator = CompetingRisksGenerator(
        ...     event_types=['progression', 'death', 'dropout'],
        ...     event_weights=[0.5, 0.3, 0.2]
        ... )
        >>> outcome = generator.generate(sequences)
    """
    
    def __init__(
        self,
        event_types: List[str],
        event_weights: Optional[List[float]] = None,
        censoring_rate: float = 0.2,
        seed: Optional[int] = 42,
    ):
        """
        Initialize competing risks generator.
        
        Args:
            event_types: List of event type names
            event_weights: Relative probabilities for each event type
                          If None, uses uniform weights
            censoring_rate: Proportion of patients to censor
            seed: Random seed
        """
        self.event_types = event_types
        self.num_event_types = len(event_types)
        
        if event_weights is None:
            self.event_weights = [1.0 / self.num_event_types] * self.num_event_types
        else:
            # Normalize weights
            total = sum(event_weights)
            self.event_weights = [w / total for w in event_weights]
        
        self.censoring_rate = censoring_rate
        self.seed = seed
    
    def generate(self, sequences: List[PatientSequence]) -> Dict[str, torch.Tensor]:
        """
        Generate competing risks outcomes.
        
        Args:
            sequences: List of PatientSequence objects
        
        Returns:
            Dictionary with:
                - event_times: Event or censoring times
                - event_types: Event type indices (0 = censored, 1+ = event types)
                - event_indicators: 1 if any event observed, 0 if censored
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        
        event_times = []
        event_type_indices = []
        event_indicators = []
        
        # Use discrete-time generator as base
        base_generator = DiscreteTimeSurvivalGenerator(
            censoring_rate=0.0,  # Handle censoring separately
            seed=None
        )
        
        for seq in sequences:
            # Compute risk score
            risk_factors = base_generator._compute_risk_factors(seq)
            risk_score = base_generator._compute_risk_score(risk_factors)
            
            # Simulate event time
            event_time = base_generator._simulate_event_time(risk_score, len(seq.visits))
            
            # Apply censoring
            is_censored = np.random.random() < self.censoring_rate
            
            if is_censored:
                event_times.append(np.random.randint(0, len(seq.visits)))
                event_type_indices.append(0)  # 0 = censored
                event_indicators.append(0)
            else:
                # Choose event type based on weights
                event_type_idx = np.random.choice(self.num_event_types, p=self.event_weights)
                event_times.append(event_time)
                event_type_indices.append(event_type_idx + 1)  # 1+ = event types
                event_indicators.append(1)
        
        return {
            'event_times': torch.tensor(event_times),
            'event_types': torch.tensor(event_type_indices),
            'event_indicators': torch.tensor(event_indicators),
            'metadata': {
                'event_type_names': self.event_types,
                'event_weights': self.event_weights,
                'censoring_rate': self.censoring_rate,
            }
        }
