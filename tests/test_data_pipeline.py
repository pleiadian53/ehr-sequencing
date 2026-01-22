"""
Unit tests for data pipeline components.

Tests:
- BaseEHRAdapter and SyntheaAdapter
- VisitGrouper with different strategies
- PatientSequenceBuilder
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import pandas as pd

from ehrsequencing.data import (
    MedicalEvent,
    PatientInfo,
    SyntheaAdapter,
    Visit,
    VisitGrouper,
    PatientSequence,
    PatientSequenceBuilder
)


class TestMedicalEvent:
    """Test MedicalEvent dataclass."""
    
    def test_create_valid_event(self):
        """Test creating a valid medical event."""
        event = MedicalEvent(
            patient_id='patient_1',
            timestamp=datetime(2020, 1, 15, 9, 0),
            code='E11.9',
            code_type='diagnosis',
            code_system='ICD10'
        )
        
        assert event.patient_id == 'patient_1'
        assert event.code == 'E11.9'
        assert event.code_type == 'diagnosis'
    
    def test_event_requires_patient_id(self):
        """Test that patient_id is required."""
        with pytest.raises(ValueError):
            MedicalEvent(
                patient_id='',
                timestamp=datetime.now(),
                code='E11.9',
                code_type='diagnosis',
                code_system='ICD10'
            )
    
    def test_event_requires_datetime(self):
        """Test that timestamp must be datetime."""
        with pytest.raises(TypeError):
            MedicalEvent(
                patient_id='patient_1',
                timestamp='2020-01-15',
                code='E11.9',
                code_type='diagnosis',
                code_system='ICD10'
            )


class TestVisitGrouper:
    """Test VisitGrouper functionality."""
    
    @pytest.fixture
    def sample_events(self):
        """Create sample events for testing."""
        base_time = datetime(2020, 1, 15, 9, 0)
        
        events = [
            MedicalEvent(
                patient_id='patient_1',
                timestamp=base_time,
                code='E11.9',
                code_type='diagnosis',
                code_system='ICD10',
                encounter_id='enc_1'
            ),
            MedicalEvent(
                patient_id='patient_1',
                timestamp=base_time + timedelta(hours=1),
                code='4548-4',
                code_type='lab',
                code_system='LOINC',
                encounter_id='enc_1'
            ),
            MedicalEvent(
                patient_id='patient_1',
                timestamp=base_time + timedelta(hours=2),
                code='860975',
                code_type='medication',
                code_system='RXNORM',
                encounter_id='enc_1'
            ),
            # Different encounter
            MedicalEvent(
                patient_id='patient_1',
                timestamp=base_time + timedelta(days=7),
                code='I10',
                code_type='diagnosis',
                code_system='ICD10',
                encounter_id='enc_2'
            )
        ]
        
        return events
    
    def test_group_by_encounter(self, sample_events):
        """Test grouping by encounter ID."""
        grouper = VisitGrouper(strategy='encounter')
        visits = grouper.group_events(sample_events)
        
        assert len(visits) == 2
        assert visits[0].encounter_id == 'enc_1'
        assert visits[0].num_codes() == 3
        assert visits[1].encounter_id == 'enc_2'
        assert visits[1].num_codes() == 1
    
    def test_group_by_same_day(self, sample_events):
        """Test grouping by same calendar day."""
        grouper = VisitGrouper(strategy='same_day')
        visits = grouper.group_events(sample_events)
        
        assert len(visits) == 2
        assert visits[0].num_codes() == 3  # First 3 events on same day
        assert visits[1].num_codes() == 1  # Last event 7 days later
    
    def test_semantic_code_ordering(self, sample_events):
        """Test that codes are ordered semantically within visits."""
        grouper = VisitGrouper(strategy='encounter', preserve_code_types=True)
        visits = grouper.group_events(sample_events)
        
        # First visit should have codes grouped by type
        visit = visits[0]
        assert 'diagnosis' in visit.codes_by_type
        assert 'lab' in visit.codes_by_type
        assert 'medication' in visit.codes_by_type
        
        # Get ordered codes (diagnosis, lab, medication)
        ordered = visit.get_ordered_codes()
        assert ordered[0] == 'E11.9'  # diagnosis first
        assert ordered[1] == '4548-4'  # lab second
        assert ordered[2] == '860975'  # medication third
    
    def test_hybrid_strategy(self, sample_events):
        """Test hybrid grouping strategy."""
        # Add event without encounter ID
        events_with_missing = sample_events + [
            MedicalEvent(
                patient_id='patient_1',
                timestamp=datetime(2020, 1, 20, 10, 0),
                code='Z00.00',
                code_type='diagnosis',
                code_system='ICD10',
                encounter_id=None
            )
        ]
        
        grouper = VisitGrouper(strategy='hybrid')
        visits = grouper.group_events(events_with_missing)
        
        # Should have 3 visits: 2 with encounter IDs, 1 without
        assert len(visits) == 3


class TestPatientSequenceBuilder:
    """Test PatientSequenceBuilder functionality."""
    
    @pytest.fixture
    def sample_visits(self):
        """Create sample visits for testing."""
        base_time = datetime(2020, 1, 1)
        
        visits = [
            Visit(
                visit_id='visit_1',
                patient_id='patient_1',
                timestamp=base_time,
                codes_by_type={'diagnosis': ['E11.9'], 'lab': ['4548-4']},
                codes_flat=['E11.9', '4548-4']
            ),
            Visit(
                visit_id='visit_2',
                patient_id='patient_1',
                timestamp=base_time + timedelta(days=30),
                codes_by_type={'diagnosis': ['I10'], 'medication': ['860975']},
                codes_flat=['I10', '860975']
            ),
            Visit(
                visit_id='visit_3',
                patient_id='patient_1',
                timestamp=base_time + timedelta(days=60),
                codes_by_type={'lab': ['2345-7']},
                codes_flat=['2345-7']
            )
        ]
        
        return visits
    
    def test_build_vocabulary(self, sample_visits):
        """Test vocabulary building."""
        patient_visits = {'patient_1': sample_visits}
        
        builder = PatientSequenceBuilder()
        vocab = builder.build_vocabulary(patient_visits, min_frequency=1)
        
        # Should have special tokens + all codes
        assert '[PAD]' in vocab
        assert '[UNK]' in vocab
        assert 'E11.9' in vocab
        assert '4548-4' in vocab
        assert len(vocab) >= 5  # Special tokens + codes
    
    def test_build_sequences(self, sample_visits):
        """Test building patient sequences."""
        patient_visits = {'patient_1': sample_visits}
        
        builder = PatientSequenceBuilder()
        sequences = builder.build_sequences(patient_visits, min_visits=2)
        
        assert len(sequences) == 1
        sequence = sequences[0]
        assert sequence.patient_id == 'patient_1'
        assert sequence.sequence_length == 3
        assert len(sequence.visits) == 3
    
    def test_encode_sequence(self, sample_visits):
        """Test encoding a patient sequence."""
        patient_visits = {'patient_1': sample_visits}
        
        builder = PatientSequenceBuilder(max_visits=10, max_codes_per_visit=5)
        builder.build_vocabulary(patient_visits)
        
        sequences = builder.build_sequences(patient_visits)
        encoded = builder.encode_sequence(sequences[0], return_tensors=False)
        
        assert 'visit_codes' in encoded
        assert 'visit_mask' in encoded
        assert 'sequence_mask' in encoded
        assert 'time_deltas' in encoded
        
        # Check shapes
        assert len(encoded['visit_codes']) == 10  # max_visits
        assert len(encoded['visit_codes'][0]) == 5  # max_codes_per_visit
    
    def test_time_deltas(self, sample_visits):
        """Test time delta calculation."""
        sequence = PatientSequence(
            patient_id='patient_1',
            visits=sample_visits,
            sequence_length=len(sample_visits)
        )
        
        deltas = sequence.get_time_deltas()
        
        assert len(deltas) == 2  # 3 visits = 2 deltas
        assert deltas[0] == 30.0  # 30 days between visit 1 and 2
        assert deltas[1] == 30.0  # 30 days between visit 2 and 3
    
    def test_create_dataset(self, sample_visits):
        """Test creating PyTorch dataset."""
        patient_visits = {'patient_1': sample_visits}
        
        builder = PatientSequenceBuilder()
        builder.build_vocabulary(patient_visits)
        sequences = builder.build_sequences(patient_visits)
        
        dataset = builder.create_dataset(sequences)
        
        assert len(dataset) == 1
        
        # Get first item
        item = dataset[0]
        assert 'visit_codes' in item
        assert 'patient_id' in item


class TestSyntheaAdapter:
    """Test SyntheaAdapter (requires mock data)."""
    
    @pytest.fixture
    def mock_synthea_dir(self):
        """Create mock Synthea data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create mock patients.csv
            patients_df = pd.DataFrame({
                'Id': ['patient_1', 'patient_2'],
                'BIRTHDATE': ['1980-01-01', '1975-05-15'],
                'GENDER': ['M', 'F'],
                'RACE': ['white', 'black'],
                'ETHNICITY': ['nonhispanic', 'hispanic'],
                'CITY': ['Boston', 'Cambridge'],
                'STATE': ['MA', 'MA']
            })
            patients_df.to_csv(tmpdir / 'patients.csv', index=False)
            
            # Create mock encounters.csv
            encounters_df = pd.DataFrame({
                'Id': ['enc_1', 'enc_2'],
                'PATIENT': ['patient_1', 'patient_1'],
                'START': ['2020-01-15 09:00:00', '2020-02-20 10:00:00'],
                'STOP': ['2020-01-15 10:00:00', '2020-02-20 11:00:00'],
                'ENCOUNTERCLASS': ['outpatient', 'outpatient']
            })
            encounters_df.to_csv(tmpdir / 'encounters.csv', index=False)
            
            # Create mock conditions.csv
            conditions_df = pd.DataFrame({
                'PATIENT': ['patient_1', 'patient_1'],
                'ENCOUNTER': ['enc_1', 'enc_2'],
                'CODE': ['44054006', '38341003'],
                'DESCRIPTION': ['Type 2 Diabetes', 'Hypertension'],
                'START': ['2020-01-15', '2020-02-20']
            })
            conditions_df.to_csv(tmpdir / 'conditions.csv', index=False)
            
            # Create empty observations.csv with headers
            observations_df = pd.DataFrame(columns=[
                'PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION', 'DATE', 
                'VALUE', 'UNITS', 'TYPE'
            ])
            observations_df.to_csv(tmpdir / 'observations.csv', index=False)
            
            # Create empty medications.csv with headers
            medications_df = pd.DataFrame(columns=[
                'PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION', 'START',
                'STOP', 'REASONCODE'
            ])
            medications_df.to_csv(tmpdir / 'medications.csv', index=False)
            
            # Create empty procedures.csv with headers
            procedures_df = pd.DataFrame(columns=[
                'PATIENT', 'ENCOUNTER', 'CODE', 'DESCRIPTION', 'DATE',
                'REASONCODE'
            ])
            procedures_df.to_csv(tmpdir / 'procedures.csv', index=False)
            
            yield tmpdir
    
    def test_adapter_initialization(self, mock_synthea_dir):
        """Test adapter initialization with valid directory."""
        adapter = SyntheaAdapter(str(mock_synthea_dir))
        assert adapter.data_dir == mock_synthea_dir
    
    def test_load_patients(self, mock_synthea_dir):
        """Test loading patient demographics."""
        adapter = SyntheaAdapter(str(mock_synthea_dir))
        patients = adapter.load_patients()
        
        assert len(patients) == 2
        assert patients[0].patient_id == 'patient_1'
        assert patients[0].gender == 'M'
    
    def test_load_events(self, mock_synthea_dir):
        """Test loading medical events."""
        adapter = SyntheaAdapter(str(mock_synthea_dir))
        events = adapter.load_events()
        
        assert len(events) >= 2  # At least 2 conditions
        assert all(isinstance(e, MedicalEvent) for e in events)
    
    def test_get_statistics(self, mock_synthea_dir):
        """Test getting dataset statistics."""
        adapter = SyntheaAdapter(str(mock_synthea_dir))
        stats = adapter.get_statistics()
        
        assert 'num_patients' in stats
        assert 'num_events' in stats
        assert stats['num_patients'] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
