"""
Base adapter interface for EHR data sources.

This module defines the abstract base class for all EHR data adapters.
Adapters transform raw EHR data from various sources (Synthea, MIMIC, etc.)
into a standardized event stream format.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, Iterator
import pandas as pd


@dataclass
class MedicalEvent:
    """
    Standardized representation of a single medical event.
    
    Attributes:
        patient_id: Unique patient identifier
        timestamp: Event timestamp (datetime)
        code: Medical code (ICD, LOINC, RxNorm, CPT, etc.)
        code_type: Type of code (diagnosis, lab, medication, procedure)
        code_system: Coding system (ICD9, ICD10, LOINC, RXNORM, CPT)
        value: Optional value (for labs, vitals)
        unit: Optional unit (for labs, vitals)
        encounter_id: Optional encounter/visit identifier
        metadata: Additional metadata
    """
    patient_id: str
    timestamp: datetime
    code: str
    code_type: str  # diagnosis, lab, medication, procedure, vital
    code_system: str  # ICD9, ICD10, LOINC, RXNORM, CPT, etc.
    value: Optional[float] = None
    unit: Optional[str] = None
    encounter_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate event after initialization."""
        if not self.patient_id:
            raise ValueError("patient_id cannot be empty")
        if not self.code:
            raise ValueError("code cannot be empty")
        if not self.code_type:
            raise ValueError("code_type cannot be empty")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime object")


@dataclass
class PatientInfo:
    """
    Patient demographic and metadata.
    
    Attributes:
        patient_id: Unique patient identifier
        birth_date: Date of birth
        gender: Gender (M/F/Other)
        race: Race/ethnicity
        death_date: Date of death (if applicable)
        metadata: Additional metadata
    """
    patient_id: str
    birth_date: Optional[datetime] = None
    gender: Optional[str] = None
    race: Optional[str] = None
    death_date: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseEHRAdapter(ABC):
    """
    Abstract base class for EHR data adapters.
    
    All adapters must implement methods to:
    1. Load patient demographics
    2. Load medical events
    3. Provide data statistics
    
    Example:
        >>> adapter = SyntheaAdapter(data_path='data/synthea/')
        >>> patients = adapter.load_patients()
        >>> events = adapter.load_events(patient_ids=['patient_1'])
        >>> stats = adapter.get_statistics()
    """
    
    def __init__(self, data_path: str, **kwargs):
        """
        Initialize adapter.
        
        Args:
            data_path: Path to data directory or database connection string
            **kwargs: Additional adapter-specific parameters
        """
        self.data_path = data_path
        self.config = kwargs
        self._validate_data_path()
    
    @abstractmethod
    def _validate_data_path(self) -> None:
        """
        Validate that data path exists and contains required files.
        
        Raises:
            FileNotFoundError: If data path or required files don't exist
            ValueError: If data path is invalid
        """
        pass
    
    @abstractmethod
    def load_patients(
        self,
        patient_ids: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[PatientInfo]:
        """
        Load patient demographics.
        
        Args:
            patient_ids: Optional list of patient IDs to load (None = all)
            limit: Optional limit on number of patients
        
        Returns:
            List of PatientInfo objects
        """
        pass
    
    @abstractmethod
    def load_events(
        self,
        patient_ids: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        code_types: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[MedicalEvent]:
        """
        Load medical events.
        
        Args:
            patient_ids: Optional list of patient IDs (None = all)
            start_date: Optional start date filter
            end_date: Optional end date filter
            code_types: Optional list of code types to include
            limit: Optional limit on number of events
        
        Returns:
            List of MedicalEvent objects, sorted by (patient_id, timestamp)
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with statistics:
                - num_patients: Total number of patients
                - num_events: Total number of events
                - date_range: (min_date, max_date)
                - code_types: Distribution of code types
                - code_systems: Distribution of code systems
                - events_per_patient: Mean, median, std
        """
        pass
    
    def iter_patients(
        self,
        batch_size: int = 100,
        **kwargs
    ) -> Iterator[List[PatientInfo]]:
        """
        Iterate over patients in batches.
        
        Args:
            batch_size: Number of patients per batch
            **kwargs: Additional arguments passed to load_patients
        
        Yields:
            Batches of PatientInfo objects
        """
        all_patients = self.load_patients(**kwargs)
        for i in range(0, len(all_patients), batch_size):
            yield all_patients[i:i + batch_size]
    
    def iter_events(
        self,
        batch_size: int = 1000,
        **kwargs
    ) -> Iterator[List[MedicalEvent]]:
        """
        Iterate over events in batches.
        
        Args:
            batch_size: Number of events per batch
            **kwargs: Additional arguments passed to load_events
        
        Yields:
            Batches of MedicalEvent objects
        """
        all_events = self.load_events(**kwargs)
        for i in range(0, len(all_events), batch_size):
            yield all_events[i:i + batch_size]
    
    def get_patient_timeline(
        self,
        patient_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get complete timeline for a single patient.
        
        Args:
            patient_id: Patient identifier
            start_date: Optional start date filter
            end_date: Optional end date filter
        
        Returns:
            DataFrame with columns: timestamp, code, code_type, code_system,
                                   value, unit, encounter_id
        """
        events = self.load_events(
            patient_ids=[patient_id],
            start_date=start_date,
            end_date=end_date
        )
        
        if not events:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'timestamp': e.timestamp,
                'code': e.code,
                'code_type': e.code_type,
                'code_system': e.code_system,
                'value': e.value,
                'unit': e.unit,
                'encounter_id': e.encounter_id
            }
            for e in events
        ]).sort_values('timestamp')
    
    def validate_events(self, events: List[MedicalEvent]) -> Dict[str, Any]:
        """
        Validate a list of medical events.
        
        Args:
            events: List of MedicalEvent objects
        
        Returns:
            Dictionary with validation results:
                - valid: Number of valid events
                - invalid: Number of invalid events
                - errors: List of error messages
        """
        valid = 0
        invalid = 0
        errors = []
        
        for i, event in enumerate(events):
            try:
                # Check required fields
                if not event.patient_id:
                    raise ValueError(f"Event {i}: Missing patient_id")
                if not event.code:
                    raise ValueError(f"Event {i}: Missing code")
                if not event.timestamp:
                    raise ValueError(f"Event {i}: Missing timestamp")
                
                # Check timestamp is valid
                if not isinstance(event.timestamp, datetime):
                    raise TypeError(f"Event {i}: timestamp must be datetime")
                
                valid += 1
            except (ValueError, TypeError) as e:
                invalid += 1
                errors.append(str(e))
        
        return {
            'valid': valid,
            'invalid': invalid,
            'errors': errors[:100]  # Limit to first 100 errors
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(data_path='{self.data_path}')"
