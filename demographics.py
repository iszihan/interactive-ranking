"""Lightweight demographics schema and helpers shared across servers.

All fields are optional to avoid blocking users; callers decide what to
collect in the UI. Multi-select questions use lists; single-choice
questions use strings matching the displayed label.
"""

from __future__ import annotations

import time
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class Demographics(BaseModel):
	# Single-choice questions
	age_group: Optional[str] = None
	gender_identity: Optional[str] = None
	gender_self_describe: Optional[str] = None
	familiarity: Optional[str] = None
	usage_frequency: Optional[str] = None

	# Multi-select questions
	experience_depth: List[str] = Field(default_factory=list)
	domain_background: List[str] = Field(default_factory=list)

	# Optional linkage to session/user if the client supplies it
	participant_id: Optional[str] = None

	# Timestamp is filled on the server when the record is received
	timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

	@validator(
		"age_group",
		"gender_identity",
		"gender_self_describe",
		"familiarity",
		"usage_frequency",
		pre=True,
	)
	def _strip_strings(cls, value: Optional[str]) -> Optional[str]:  # noqa: N805
		if value is None:
			return None
		value = str(value).strip()
		return value or None

	@validator("experience_depth", "domain_background", pre=True)
	def _normalize_list(cls, value):  # noqa: N805
		if value is None:
			return []
		if isinstance(value, (str, bytes)):
			return [str(value).strip()] if str(value).strip() else []
		try:
			return [str(v).strip() for v in list(value) if str(v).strip()]
		except Exception:
			return []

	def to_storage_dict(self) -> dict:
		"""Return a JSON-friendly dict with stable ordering."""
		return {
			"timestamp_ms": self.timestamp_ms,
			"participant_id": self.participant_id,
			"age_group": self.age_group,
			"gender_identity": self.gender_identity,
			"gender_self_describe": self.gender_self_describe,
			"familiarity": self.familiarity,
			"usage_frequency": self.usage_frequency,
			"experience_depth": list(self.experience_depth or []),
			"domain_background": list(self.domain_background or []),
		}


__all__ = ["Demographics"]
