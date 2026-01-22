"""
Visit grouper for organizing medical events into visits.

This module groups medical events into visits based on encounter IDs or
temporal proximity. It also handles within-visit code ordering using
semantic grouping by code type.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging

from .adapters.base import MedicalEvent

logger = logging.getLogger(__name__)


@dataclass
class Visit:
    """
    Representation of a single visit with grouped medical codes.
    
    Attributes:
        visit_id: Unique visit identifier
        patient_id: Patient identifier
        timestamp: Visit date/time (start of visit)
        encounter_id: Optional encounter ID from EHR
        codes_by_type: Dictionary mapping code type to list of codes
        codes_flat: Flat list of all codes (for simple models)
        metadata: Additional visit metadata
    """
    visit_id: str
    patient_id: str
    timestamp: datetime
    encounter_id: Optional[str] = None
    codes_by_type: Optional[Dict[str, List[str]]] = None
    codes_flat: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        """Initialize codes structures if not provided."""
        if self.codes_by_type is None:
            self.codes_by_type = {}
        if self.codes_flat is None:
            self.codes_flat = []
    
    def get_all_codes(self) -> List[str]:
        """Get all codes in the visit (flattened)."""
        if self.codes_flat:
            return self.codes_flat
        
        all_codes = []
        for codes in self.codes_by_type.values():
            all_codes.extend(codes)
        return all_codes
    
    def get_ordered_codes(self, type_order: Optional[List[str]] = None) -> List[str]:
        """
        Get codes ordered by type (semantic ordering).
        
        Args:
            type_order: Optional custom ordering of code types
                       Default: ['diagnosis', 'lab', 'vital', 'procedure', 'medication']
        
        Returns:
            List of codes ordered by type
        """
        if type_order is None:
            type_order = ['diagnosis', 'lab', 'vital', 'procedure', 'medication']
        
        ordered_codes = []
        for code_type in type_order:
            if code_type in self.codes_by_type:
                ordered_codes.extend(self.codes_by_type[code_type])
        
        # Add any remaining types not in the order
        for code_type, codes in self.codes_by_type.items():
            if code_type not in type_order:
                ordered_codes.extend(codes)
        
        return ordered_codes
    
    def num_codes(self) -> int:
        """Get total number of codes in visit."""
        return len(self.get_all_codes())


class VisitGrouper:
    """
    Groups medical events into visits.
    
    Supports two grouping strategies:
    1. Encounter-based: Use explicit encounter IDs from EHR
    2. Time-based: Group events within same day or time window
    
    Within each visit, codes are organized by type (diagnosis, lab, etc.)
    to preserve semantic structure.
    
    Example:
        >>> grouper = VisitGrouper(strategy='encounter', fallback='same_day')
        >>> events = adapter.load_events(patient_ids=['patient_1'])
        >>> visits = grouper.group_events(events)
        >>> print(f"Grouped {len(events)} events into {len(visits)} visits")
    """
    
    def __init__(
        self,
        strategy: str = 'hybrid',
        time_window_hours: int = 24,
        preserve_code_types: bool = True,
        code_type_order: Optional[List[str]] = None
    ):
        """
        Initialize visit grouper.
        
        Args:
            strategy: Grouping strategy
                     - 'encounter': Use encounter IDs only
                     - 'same_day': Group events on same calendar day
                     - 'time_window': Group events within time window
                     - 'hybrid': Use encounter IDs when available, fallback to same_day
            time_window_hours: Hours for time window grouping (default: 24)
            preserve_code_types: Whether to preserve code type structure (default: True)
            code_type_order: Order of code types for semantic ordering
        """
        if strategy not in ['encounter', 'same_day', 'time_window', 'hybrid']:
            raise ValueError(f"Invalid strategy: {strategy}")
        
        self.strategy = strategy
        self.time_window_hours = time_window_hours
        self.preserve_code_types = preserve_code_types
        self.code_type_order = code_type_order or [
            'diagnosis', 'lab', 'vital', 'procedure', 'medication'
        ]
        
        logger.info(f"Initialized VisitGrouper with strategy='{strategy}'")
    
    def group_events(
        self,
        events: List[MedicalEvent],
        patient_id: Optional[str] = None
    ) -> List[Visit]:
        """
        Group medical events into visits.
        
        Args:
            events: List of MedicalEvent objects (should be for single patient)
            patient_id: Optional patient ID (for validation)
        
        Returns:
            List of Visit objects, sorted by timestamp
        """
        if not events:
            return []
        
        # Validate single patient
        patient_ids = set(e.patient_id for e in events)
        if len(patient_ids) > 1:
            if patient_id:
                events = [e for e in events if e.patient_id == patient_id]
            else:
                logger.warning(f"Events contain {len(patient_ids)} patients. Grouping all together.")
        
        # Sort events by timestamp
        events = sorted(events, key=lambda e: e.timestamp)
        
        # Group based on strategy
        if self.strategy == 'encounter':
            visits = self._group_by_encounter(events)
        elif self.strategy == 'same_day':
            visits = self._group_by_same_day(events)
        elif self.strategy == 'time_window':
            visits = self._group_by_time_window(events)
        elif self.strategy == 'hybrid':
            visits = self._group_hybrid(events)
        
        logger.info(f"Grouped {len(events)} events into {len(visits)} visits")
        return visits
    
    def _group_by_encounter(self, events: List[MedicalEvent]) -> List[Visit]:
        """Group events by encounter ID."""
        encounter_groups = defaultdict(list)
        
        for event in events:
            if event.encounter_id:
                encounter_groups[event.encounter_id].append(event)
            else:
                # Create single-event visit for events without encounter ID
                encounter_groups[f"no_encounter_{id(event)}"].append(event)
        
        visits = []
        for encounter_id, group_events in encounter_groups.items():
            visit = self._create_visit(group_events, encounter_id)
            visits.append(visit)
        
        return sorted(visits, key=lambda v: v.timestamp)
    
    def _group_by_same_day(self, events: List[MedicalEvent]) -> List[Visit]:
        """Group events that occur on the same calendar day."""
        day_groups = defaultdict(list)
        
        for event in events:
            day_key = (event.patient_id, event.timestamp.date())
            day_groups[day_key].append(event)
        
        visits = []
        for (patient_id, date), group_events in day_groups.items():
            visit_id = f"{patient_id}_{date.isoformat()}"
            visit = self._create_visit(group_events, visit_id)
            visits.append(visit)
        
        return sorted(visits, key=lambda v: v.timestamp)
    
    def _group_by_time_window(self, events: List[MedicalEvent]) -> List[Visit]:
        """Group events within a time window."""
        if not events:
            return []
        
        visits = []
        current_group = [events[0]]
        
        for event in events[1:]:
            time_diff = (event.timestamp - current_group[0].timestamp).total_seconds() / 3600
            
            if time_diff <= self.time_window_hours:
                current_group.append(event)
            else:
                # Create visit from current group
                visit = self._create_visit(current_group)
                visits.append(visit)
                current_group = [event]
        
        # Create visit from last group
        if current_group:
            visit = self._create_visit(current_group)
            visits.append(visit)
        
        return visits
    
    def _group_hybrid(self, events: List[MedicalEvent]) -> List[Visit]:
        """
        Hybrid grouping: use encounter IDs when available, fallback to same-day.
        """
        # Separate events with and without encounter IDs
        with_encounter = [e for e in events if e.encounter_id]
        without_encounter = [e for e in events if not e.encounter_id]
        
        # Group events with encounter IDs
        visits = []
        if with_encounter:
            encounter_groups = defaultdict(list)
            for event in with_encounter:
                encounter_groups[event.encounter_id].append(event)
            
            for encounter_id, group_events in encounter_groups.items():
                visit = self._create_visit(group_events, encounter_id)
                visits.append(visit)
        
        # Group events without encounter IDs by same day
        if without_encounter:
            day_groups = defaultdict(list)
            for event in without_encounter:
                day_key = (event.patient_id, event.timestamp.date())
                day_groups[day_key].append(event)
            
            for (patient_id, date), group_events in day_groups.items():
                visit_id = f"{patient_id}_{date.isoformat()}_no_encounter"
                visit = self._create_visit(group_events, visit_id)
                visits.append(visit)
        
        return sorted(visits, key=lambda v: v.timestamp)
    
    def _create_visit(
        self,
        events: List[MedicalEvent],
        visit_id: Optional[str] = None
    ) -> Visit:
        """
        Create a Visit object from a group of events.
        
        Args:
            events: List of events in this visit
            visit_id: Optional visit ID (generated if not provided)
        
        Returns:
            Visit object
        """
        if not events:
            raise ValueError("Cannot create visit from empty event list")
        
        # Sort events by timestamp
        events = sorted(events, key=lambda e: e.timestamp)
        
        # Get basic info
        patient_id = events[0].patient_id
        timestamp = events[0].timestamp
        encounter_id = events[0].encounter_id
        
        # Generate visit ID if not provided
        if visit_id is None:
            if encounter_id:
                visit_id = encounter_id
            else:
                visit_id = f"{patient_id}_{timestamp.isoformat()}"
        
        # Group codes by type
        codes_by_type = defaultdict(list)
        codes_flat = []
        
        for event in events:
            code = event.code
            codes_flat.append(code)
            
            if self.preserve_code_types:
                codes_by_type[event.code_type].append(code)
        
        # Create visit
        visit = Visit(
            visit_id=visit_id,
            patient_id=patient_id,
            timestamp=timestamp,
            encounter_id=encounter_id,
            codes_by_type=dict(codes_by_type) if self.preserve_code_types else None,
            codes_flat=codes_flat,
            metadata={
                'num_events': len(events),
                'code_types': list(codes_by_type.keys()) if self.preserve_code_types else None,
                'duration': (events[-1].timestamp - events[0].timestamp).total_seconds() / 3600  # hours
            }
        )
        
        return visit
    
    def group_by_patient(
        self,
        events: List[MedicalEvent]
    ) -> Dict[str, List[Visit]]:
        """
        Group events into visits for multiple patients.
        
        Args:
            events: List of MedicalEvent objects (can be for multiple patients)
        
        Returns:
            Dictionary mapping patient_id to list of Visit objects
        """
        # Group events by patient
        patient_events = defaultdict(list)
        for event in events:
            patient_events[event.patient_id].append(event)
        
        # Group each patient's events into visits
        patient_visits = {}
        for patient_id, p_events in patient_events.items():
            visits = self.group_events(p_events, patient_id=patient_id)
            patient_visits[patient_id] = visits
        
        logger.info(f"Grouped events for {len(patient_visits)} patients")
        return patient_visits
