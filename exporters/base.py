"""Shared lifecycle for session-bound exporters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import ClassVar

from ..core import Collection, IBLDataset, OneSessionConverter


class Exporter(ABC):
    """Bind an exporter to one session and provide its shared lifecycle."""

    settings: ClassVar[dict[str, object]] = {}
    datasets: ClassVar[dict[str, IBLDataset]] = {}

    def __init__(self, session_converter: OneSessionConverter) -> None:
        self.session_converter = session_converter
        for name, default in self.settings.items():
            setattr(self, name, deepcopy(default))

    def __repr__(self) -> str:
        settings = ", ".join(
            f"{name}={getattr(self, name)!r}" for name in self.settings
        )
        return f"{type(self).__name__}({settings})"

    @abstractmethod
    def get_collections(self) -> list[Collection]:
        """Return the collections declared by the concrete exporter."""
