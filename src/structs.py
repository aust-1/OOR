"""Data structures for nurse scheduling problem."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Nurse:
    nurse_id: int
    name: str
    skill_level: str
    contract: str
    absence_prob: float


@dataclass
class CoverageRequirement:
    day: str
    shift: str
    required_total: int
    required_senior: int
    required_icu: int


@dataclass
class Preference:
    nurse_id: int
    day: str
    shift: str
    preference_type: str  # "want" or "avoid"
    weight: int
