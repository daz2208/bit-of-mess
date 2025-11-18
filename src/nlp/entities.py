"""Entity extraction from natural language."""

from typing import Any
import re
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class Entity:
    """Extracted entity."""
    type: str
    value: str
    normalized: Any
    start: int
    end: int
    confidence: float


class EntityExtractor:
    """
    Extract entities from natural language text.

    Extracts: dates, times, durations, people, locations, numbers, etc.
    """

    def __init__(self):
        # Time patterns
        self.time_patterns = [
            (r'\b(\d{1,2}):(\d{2})\s*(am|pm)?\b', 'absolute'),
            (r'\b(\d{1,2})\s*(am|pm)\b', 'hour_only'),
            (r'\b(noon|midnight)\b', 'named'),
            (r'\bin\s+(\d+)\s+(minute|hour|day|week)s?\b', 'relative'),
            (r'\b(morning|afternoon|evening|night)\b', 'period')
        ]

        # Date patterns
        self.date_patterns = [
            (r'\b(today|tomorrow|yesterday)\b', 'relative_day'),
            (r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', 'weekday'),
            (r'\bnext\s+(week|month|year)\b', 'relative_period'),
            (r'\b(\d{1,2})[/\-](\d{1,2})[/\-]?(\d{2,4})?\b', 'numeric'),
            (r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+(\d{1,2})\b', 'month_day')
        ]

        # Duration patterns
        self.duration_patterns = [
            r'\bfor\s+(\d+)\s*(minute|hour|day|week|month)s?\b',
            r'\b(\d+)\s*(min|hr|h|m)\b',
            r'\b(half\s+an?\s+hour|quarter\s+hour)\b'
        ]

        # Person patterns
        self.person_patterns = [
            r'\bwith\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+and\s+I\b',
            r'\bcontact\s+([A-Z][a-z]+)\b',
            r'\bemail\s+([A-Z][a-z]+)\b'
        ]

        # Location patterns
        self.location_patterns = [
            r'\bat\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\bin\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\broom\s+(\w+)\b',
            r'\boffice\s+(\w+)\b'
        ]

        # Number patterns
        self.number_patterns = [
            r'\b(\d+(?:\.\d+)?)\s*(%|percent)\b',
            r'\b\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\b',
            r'\b(\d+(?:,\d{3})*)\b'
        ]

        # Priority/importance patterns
        self.priority_patterns = [
            (r'\b(urgent|asap|immediately|critical)\b', 'critical'),
            (r'\b(important|high\s+priority)\b', 'high'),
            (r'\b(normal|regular|standard)\b', 'medium'),
            (r'\b(low\s+priority|when\s+you\s+can|no\s+rush)\b', 'low')
        ]

    def extract(self, text: str) -> list[Entity]:
        """Extract all entities from text."""
        entities = []

        # Extract each entity type
        entities.extend(self._extract_times(text))
        entities.extend(self._extract_dates(text))
        entities.extend(self._extract_durations(text))
        entities.extend(self._extract_people(text))
        entities.extend(self._extract_locations(text))
        entities.extend(self._extract_numbers(text))
        entities.extend(self._extract_priorities(text))

        # Sort by position
        entities.sort(key=lambda e: e.start)

        return entities

    def extract_as_dict(self, text: str) -> dict:
        """Extract entities and return as categorized dict."""
        entities = self.extract(text)

        result = {
            "times": [],
            "dates": [],
            "durations": [],
            "people": [],
            "locations": [],
            "numbers": [],
            "priorities": []
        }

        for entity in entities:
            category = entity.type.split("_")[0] + "s"
            if category in result:
                result[category].append({
                    "value": entity.value,
                    "normalized": entity.normalized,
                    "confidence": entity.confidence
                })
            elif entity.type == "priority":
                result["priorities"].append({
                    "value": entity.value,
                    "normalized": entity.normalized,
                    "confidence": entity.confidence
                })

        return result

    def _extract_times(self, text: str) -> list[Entity]:
        """Extract time entities."""
        entities = []
        text_lower = text.lower()

        for pattern, pattern_type in self.time_patterns:
            for match in re.finditer(pattern, text_lower):
                normalized = self._normalize_time(match, pattern_type)
                entities.append(Entity(
                    type="time",
                    value=match.group(0),
                    normalized=normalized,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9 if pattern_type == 'absolute' else 0.7
                ))

        return entities

    def _normalize_time(self, match, pattern_type: str) -> str:
        """Normalize time to 24-hour format."""
        if pattern_type == 'absolute':
            hour = int(match.group(1))
            minute = int(match.group(2))
            ampm = match.group(3)
            if ampm and ampm.lower() == 'pm' and hour < 12:
                hour += 12
            elif ampm and ampm.lower() == 'am' and hour == 12:
                hour = 0
            return f"{hour:02d}:{minute:02d}"

        elif pattern_type == 'hour_only':
            hour = int(match.group(1))
            ampm = match.group(2)
            if ampm.lower() == 'pm' and hour < 12:
                hour += 12
            elif ampm.lower() == 'am' and hour == 12:
                hour = 0
            return f"{hour:02d}:00"

        elif pattern_type == 'named':
            word = match.group(1).lower()
            return "12:00" if word == "noon" else "00:00"

        elif pattern_type == 'period':
            periods = {
                "morning": "09:00",
                "afternoon": "14:00",
                "evening": "18:00",
                "night": "21:00"
            }
            return periods.get(match.group(1).lower(), "12:00")

        return match.group(0)

    def _extract_dates(self, text: str) -> list[Entity]:
        """Extract date entities."""
        entities = []
        text_lower = text.lower()
        today = datetime.now()

        for pattern, pattern_type in self.date_patterns:
            for match in re.finditer(pattern, text_lower):
                normalized = self._normalize_date(match, pattern_type, today)
                entities.append(Entity(
                    type="date",
                    value=match.group(0),
                    normalized=normalized,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85
                ))

        return entities

    def _normalize_date(self, match, pattern_type: str, today: datetime) -> str:
        """Normalize date to ISO format."""
        if pattern_type == 'relative_day':
            word = match.group(1).lower()
            if word == 'today':
                return today.strftime("%Y-%m-%d")
            elif word == 'tomorrow':
                return (today + timedelta(days=1)).strftime("%Y-%m-%d")
            elif word == 'yesterday':
                return (today - timedelta(days=1)).strftime("%Y-%m-%d")

        elif pattern_type == 'weekday':
            weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            target_day = weekdays.index(match.group(1).lower())
            current_day = today.weekday()
            days_ahead = target_day - current_day
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        return match.group(0)

    def _extract_durations(self, text: str) -> list[Entity]:
        """Extract duration entities."""
        entities = []
        text_lower = text.lower()

        for pattern in self.duration_patterns:
            for match in re.finditer(pattern, text_lower):
                entities.append(Entity(
                    type="duration",
                    value=match.group(0),
                    normalized=self._normalize_duration(match),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85
                ))

        return entities

    def _normalize_duration(self, match) -> int:
        """Normalize duration to minutes."""
        text = match.group(0).lower()

        if 'half' in text:
            return 30
        if 'quarter' in text:
            return 15

        groups = match.groups()
        if len(groups) >= 2:
            amount = int(groups[0])
            unit = groups[1].lower()

            if unit in ['minute', 'min', 'm']:
                return amount
            elif unit in ['hour', 'hr', 'h']:
                return amount * 60
            elif unit == 'day':
                return amount * 60 * 24
            elif unit == 'week':
                return amount * 60 * 24 * 7

        return 0

    def _extract_people(self, text: str) -> list[Entity]:
        """Extract person names."""
        entities = []

        for pattern in self.person_patterns:
            for match in re.finditer(pattern, text):
                entities.append(Entity(
                    type="person",
                    value=match.group(1),
                    normalized=match.group(1),
                    start=match.start(1),
                    end=match.end(1),
                    confidence=0.7
                ))

        return entities

    def _extract_locations(self, text: str) -> list[Entity]:
        """Extract location entities."""
        entities = []

        for pattern in self.location_patterns:
            for match in re.finditer(pattern, text):
                entities.append(Entity(
                    type="location",
                    value=match.group(1),
                    normalized=match.group(1),
                    start=match.start(1),
                    end=match.end(1),
                    confidence=0.65
                ))

        return entities

    def _extract_numbers(self, text: str) -> list[Entity]:
        """Extract numeric entities."""
        entities = []

        for pattern in self.number_patterns:
            for match in re.finditer(pattern, text):
                value = match.group(1).replace(',', '')
                try:
                    normalized = float(value)
                except ValueError:
                    normalized = value

                entities.append(Entity(
                    type="number",
                    value=match.group(0),
                    normalized=normalized,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9
                ))

        return entities

    def _extract_priorities(self, text: str) -> list[Entity]:
        """Extract priority/urgency entities."""
        entities = []
        text_lower = text.lower()

        for pattern, level in self.priority_patterns:
            for match in re.finditer(pattern, text_lower):
                entities.append(Entity(
                    type="priority",
                    value=match.group(0),
                    normalized=level,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85
                ))

        return entities
