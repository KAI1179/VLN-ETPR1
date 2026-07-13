"""Cognitive-map candidate contracts shared by all training stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NavigationArchitecture(str, Enum):
    CURRENT = "current"
    TRY5 = "try5"


class CognitiveMapSource(str, Enum):
    IMAGINED = "imagined"
    LLM_BOXES = "llm_boxes"
    LLM_GRID = "llm_grid"
    PRIOR_GT = "prior_gt"


@dataclass(frozen=True)
class CognitiveMapCandidate:
    architecture: NavigationArchitecture
    source: CognitiveMapSource

    def __post_init__(self) -> None:
        if (self.architecture, self.source) not in {
            (NavigationArchitecture.CURRENT, CognitiveMapSource.IMAGINED),
            (NavigationArchitecture.CURRENT, CognitiveMapSource.LLM_BOXES),
            (NavigationArchitecture.CURRENT, CognitiveMapSource.PRIOR_GT),
            (NavigationArchitecture.TRY5, CognitiveMapSource.LLM_GRID),
            (NavigationArchitecture.TRY5, CognitiveMapSource.PRIOR_GT),
        }:
            raise ValueError(
                "Unsupported cognitive-map candidate: "
                f"architecture={self.architecture.value}, source={self.source.value}"
            )

    @classmethod
    def parse(cls, architecture: str, source: str) -> "CognitiveMapCandidate":
        try:
            architecture_value = NavigationArchitecture(architecture)
        except ValueError as exc:
            raise ValueError(f"Unknown navigation architecture: {architecture}") from exc
        try:
            source_value = CognitiveMapSource(source)
        except ValueError as exc:
            raise ValueError(f"Unknown cognitive-map source: {source}") from exc
        return cls(architecture_value, source_value)

    @property
    def metadata_schema(self) -> str:
        return "direction5" if self.architecture is NavigationArchitecture.TRY5 else "path5"

    @property
    def requires_box_targets(self) -> bool:
        return self.architecture is NavigationArchitecture.CURRENT

    @property
    def uses_llm_cache(self) -> bool:
        return self.source in {
            CognitiveMapSource.LLM_BOXES,
            CognitiveMapSource.LLM_GRID,
        }

