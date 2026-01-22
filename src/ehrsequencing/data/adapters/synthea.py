"""
Synthea adapter for loading synthetic EHR data.

Synthea (https://github.com/synthetichealth/synthea) generates realistic
synthetic patient data in CSV format. This adapter loads and normalizes
Synthea data into the standardized MedicalEvent format.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

from .base import BaseEHRAdapter, MedicalEvent, PatientInfo

logger = logging.getLogger(__name__)


class SyntheaAdapter(BaseEHRAdapter):
    """
    Adapter for Synthea synthetic EHR data.
    
    Expected directory structure:
        data_path/
            ├── patients.csv
            ├── encounters.csv
            ├── conditions.csv
            ├── observations.csv
            ├── medications.csv
            └── procedures.csv
    
    Example:
        >>> adapter = SyntheaAdapter('data/synthea/')
        >>> patients = adapter.load_patients(limit=100)
        >>> events = adapter.load_events(patient_ids=[p.patient_id for p in patients])
        >>> stats = adapter.get_statistics()
    """
    
    REQUIRED_FILES = [
        'patients.csv',
        'encounters.csv',
        'conditions.csv',
        'observations.csv',
        'medications.csv',
        'procedures.csv'
    ]
    
    def __init__(self, data_path: str, **kwargs):
        """
        Initialize Synthea adapter.
        
        Args:
            data_path: Path to Synthea data directory
            **kwargs: Additional configuration options
        """
        self.data_dir = Path(data_path)
        self._cache = {}
        super().__init__(data_path, **kwargs)
    
    def _validate_data_path(self) -> None:
        """Validate that Synthea data directory exists and contains required files."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        if not self.data_dir.is_dir():
            raise ValueError(f"Data path is not a directory: {self.data_dir}")
        
        missing_files = []
        for file in self.REQUIRED_FILES:
            if not (self.data_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing required Synthea files: {', '.join(missing_files)}"
            )
        
        logger.info(f"Validated Synthea data directory: {self.data_dir}")
    
    def load_patients(
        self,
        patient_ids: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[PatientInfo]:
        """
        Load patient demographics from patients.csv.
        
        Args:
            patient_ids: Optional list of patient IDs to load
            limit: Optional limit on number of patients
        
        Returns:
            List of PatientInfo objects
        """
        df = pd.read_csv(self.data_dir / 'patients.csv')
        
        # Filter by patient IDs if provided
        if patient_ids:
            df = df[df['Id'].isin(patient_ids)]
        
        # Apply limit
        if limit:
            df = df.head(limit)
        
        patients = []
        for _, row in df.iterrows():
            patient = PatientInfo(
                patient_id=str(row['Id']),
                birth_date=pd.to_datetime(row['BIRTHDATE']) if pd.notna(row['BIRTHDATE']) else None,
                gender=str(row['GENDER']) if pd.notna(row['GENDER']) else None,
                race=str(row['RACE']) if pd.notna(row['RACE']) else None,
                death_date=pd.to_datetime(row['DEATHDATE']) if pd.notna(row.get('DEATHDATE')) else None,
                metadata={
                    'ethnicity': str(row['ETHNICITY']) if pd.notna(row.get('ETHNICITY')) else None,
                    'city': str(row['CITY']) if pd.notna(row.get('CITY')) else None,
                    'state': str(row['STATE']) if pd.notna(row.get('STATE')) else None
                }
            )
            patients.append(patient)
        
        logger.info(f"Loaded {len(patients)} patients")
        return patients
    
    def load_events(
        self,
        patient_ids: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        code_types: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[MedicalEvent]:
        """
        Load medical events from all Synthea CSV files.
        
        Args:
            patient_ids: Optional list of patient IDs
            start_date: Optional start date filter
            end_date: Optional end date filter
            code_types: Optional list of code types (diagnosis, lab, medication, procedure)
            limit: Optional limit on number of events
        
        Returns:
            List of MedicalEvent objects, sorted by (patient_id, timestamp)
        """
        all_events = []
        
        # Load conditions (diagnoses)
        if not code_types or 'diagnosis' in code_types:
            all_events.extend(self._load_conditions(patient_ids))
        
        # Load observations (labs, vitals)
        if not code_types or 'lab' in code_types or 'vital' in code_types:
            all_events.extend(self._load_observations(patient_ids))
        
        # Load medications
        if not code_types or 'medication' in code_types:
            all_events.extend(self._load_medications(patient_ids))
        
        # Load procedures
        if not code_types or 'procedure' in code_types:
            all_events.extend(self._load_procedures(patient_ids))
        
        # Filter by date range
        if start_date:
            all_events = [e for e in all_events if e.timestamp >= start_date]
        if end_date:
            all_events = [e for e in all_events if e.timestamp <= end_date]
        
        # Sort by patient and timestamp
        all_events.sort(key=lambda e: (e.patient_id, e.timestamp))
        
        # Apply limit
        if limit:
            all_events = all_events[:limit]
        
        logger.info(f"Loaded {len(all_events)} events")
        return all_events
    
    def _load_conditions(self, patient_ids: Optional[List[str]] = None) -> List[MedicalEvent]:
        """Load conditions (diagnoses) from conditions.csv."""
        df = pd.read_csv(self.data_dir / 'conditions.csv')
        
        if patient_ids:
            df = df[df['PATIENT'].isin(patient_ids)]
        
        events = []
        for _, row in df.iterrows():
            if pd.isna(row['START']):
                continue
            
            event = MedicalEvent(
                patient_id=str(row['PATIENT']),
                timestamp=pd.to_datetime(row['START']).tz_localize(None),
                code=str(row['CODE']),
                code_type='diagnosis',
                code_system='SNOMED',
                encounter_id=str(row['ENCOUNTER']) if pd.notna(row.get('ENCOUNTER')) else None,
                metadata={
                    'description': str(row['DESCRIPTION']) if pd.notna(row.get('DESCRIPTION')) else None
                }
            )
            events.append(event)
        
        return events
    
    def _load_observations(self, patient_ids: Optional[List[str]] = None) -> List[MedicalEvent]:
        """Load observations (labs, vitals) from observations.csv."""
        df = pd.read_csv(self.data_dir / 'observations.csv')
        
        if patient_ids:
            df = df[df['PATIENT'].isin(patient_ids)]
        
        events = []
        for _, row in df.iterrows():
            if pd.isna(row['DATE']):
                continue
            
            # Determine if lab or vital based on code or description
            code_type = 'lab'  # Default to lab
            if pd.notna(row.get('TYPE')):
                if 'vital' in str(row['TYPE']).lower():
                    code_type = 'vital'
            
            event = MedicalEvent(
                patient_id=str(row['PATIENT']),
                timestamp=pd.to_datetime(row['DATE']).tz_localize(None),
                code=str(row['CODE']),
                code_type=code_type,
                code_system='LOINC',
                value=float(row['VALUE']) if pd.notna(row.get('VALUE')) and str(row['VALUE']).replace('.','').replace('-','').isdigit() else None,
                unit=str(row['UNITS']) if pd.notna(row.get('UNITS')) else None,
                encounter_id=str(row['ENCOUNTER']) if pd.notna(row.get('ENCOUNTER')) else None,
                metadata={
                    'description': str(row['DESCRIPTION']) if pd.notna(row.get('DESCRIPTION')) else None,
                    'type': str(row['TYPE']) if pd.notna(row.get('TYPE')) else None
                }
            )
            events.append(event)
        
        return events
    
    def _load_medications(self, patient_ids: Optional[List[str]] = None) -> List[MedicalEvent]:
        """Load medications from medications.csv."""
        df = pd.read_csv(self.data_dir / 'medications.csv')
        
        if patient_ids:
            df = df[df['PATIENT'].isin(patient_ids)]
        
        events = []
        for _, row in df.iterrows():
            if pd.isna(row['START']):
                continue
            
            event = MedicalEvent(
                patient_id=str(row['PATIENT']),
                timestamp=pd.to_datetime(row['START']).tz_localize(None),
                code=str(row['CODE']),
                code_type='medication',
                code_system='RXNORM',
                encounter_id=str(row['ENCOUNTER']) if pd.notna(row.get('ENCOUNTER')) else None,
                metadata={
                    'description': str(row['DESCRIPTION']) if pd.notna(row.get('DESCRIPTION')) else None,
                    'stop_date': pd.to_datetime(row['STOP']).tz_localize(None) if pd.notna(row.get('STOP')) else None,
                    'reason_code': str(row['REASONCODE']) if pd.notna(row.get('REASONCODE')) else None
                }
            )
            events.append(event)
        
        return events
    
    def _load_procedures(self, patient_ids: Optional[List[str]] = None) -> List[MedicalEvent]:
        """Load procedures from procedures.csv."""
        df = pd.read_csv(self.data_dir / 'procedures.csv')
        
        if patient_ids:
            df = df[df['PATIENT'].isin(patient_ids)]
        
        events = []
        for _, row in df.iterrows():
            if pd.isna(row['START']):
                continue
            
            event = MedicalEvent(
                patient_id=str(row['PATIENT']),
                timestamp=pd.to_datetime(row['START']).tz_localize(None),
                code=str(row['CODE']),
                code_type='procedure',
                code_system='SNOMED',
                encounter_id=str(row['ENCOUNTER']) if pd.notna(row.get('ENCOUNTER')) else None,
                metadata={
                    'description': str(row['DESCRIPTION']) if pd.notna(row.get('DESCRIPTION')) else None,
                    'reason_code': str(row['REASONCODE']) if pd.notna(row.get('REASONCODE')) else None
                }
            )
            events.append(event)
        
        return events
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with comprehensive statistics
        """
        # Load all data
        patients = self.load_patients()
        events = self.load_events()
        
        # Basic counts
        num_patients = len(patients)
        num_events = len(events)
        
        # Date range
        timestamps = [e.timestamp for e in events]
        date_range = (min(timestamps), max(timestamps)) if timestamps else (None, None)
        
        # Code type distribution
        code_types = {}
        for event in events:
            code_types[event.code_type] = code_types.get(event.code_type, 0) + 1
        
        # Code system distribution
        code_systems = {}
        for event in events:
            code_systems[event.code_system] = code_systems.get(event.code_system, 0) + 1
        
        # Events per patient
        events_per_patient = {}
        for event in events:
            events_per_patient[event.patient_id] = events_per_patient.get(event.patient_id, 0) + 1
        
        event_counts = list(events_per_patient.values())
        
        stats = {
            'num_patients': num_patients,
            'num_events': num_events,
            'date_range': date_range,
            'code_types': code_types,
            'code_systems': code_systems,
            'events_per_patient': {
                'mean': sum(event_counts) / len(event_counts) if event_counts else 0,
                'median': sorted(event_counts)[len(event_counts) // 2] if event_counts else 0,
                'min': min(event_counts) if event_counts else 0,
                'max': max(event_counts) if event_counts else 0
            },
            'unique_codes': len(set(e.code for e in events)),
            'encounters_with_id': sum(1 for e in events if e.encounter_id is not None)
        }
        
        return stats
