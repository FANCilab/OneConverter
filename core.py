from __future__ import annotations

from datetime import date as Date
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Iterator

from one.alf import spec
from one.alf.path import PureALFPath


@dataclass
class IBLDataset:
    ibl_object: str
    attribute: str
    extension: str
    export_fn: Callable[[object, Path], Path]
    namespace: str | None = None
    timescale: str | None = None
    docstring: str | None = None
    extra: tuple[str, ...] = ()
    _path: PureALFPath = field(init=False, repr=False)

    def __post_init__(self) -> None:
        filename = spec.to_alf(
            self.ibl_object, self.attribute, self.extension,
            namespace=self.namespace, timescale=self.timescale, extra=self.extra,
        )
        self._path = PureALFPath(filename)
        if not self._path.is_dataset():
            raise ValueError(f"Invalid ALF dataset filename: {filename}")

    @property
    def filename(self) -> str:
        return self._path.name

    def export(self, exporter: object, output_path: Path) -> Path:
        """Export this dataset with its owning exporter to ``output_path``."""
        return self.export_fn(exporter, output_path)


@dataclass
class Collection:
    name: str
    index: int | None = None
    datasets: list[IBLDataset] = field(default_factory=list)
    collections: list[Collection] = field(default_factory=list)

    @property
    def dirname(self) -> str:
        return self.name if self.index is None else f"{self.name}_{self.index:02d}"

    def walk(self, parent: Path = Path()) -> Iterator[tuple[Path, IBLDataset]]:
        """Yield relative output paths and their datasets without opening files."""
        path = parent / self.dirname
        for dataset in sorted(self.datasets, key=lambda item: item.filename):
            yield path / dataset.filename, dataset
        for collection in sorted(self.collections, key=lambda item: item.dirname):
            yield from collection.walk(path)

    def list(self) -> list[str]:
        return [path.as_posix() for path, _ in self.walk()]


class Exporters(list[object]):
    """Session exporters, addressable by position or exporter class name."""

    def __getattr__(self, exporter_name: str) -> object:
        """Return the exporter whose class name is ``exporter_name``."""
        for exporter in self:
            if type(exporter).__name__ == exporter_name:
                return exporter
        raise AttributeError(
            f"No {exporter_name!r} exporter is available for this session"
        )


class OneSessionConverter:
    """Convert one experimental session into an IBL-compatible local session."""

    def __init__(
        self,
        modality: str,
        subject: str,
        date: str,
        session_number: int,
        *,
        raw_data_root: str | Path,
        processed_data_root: str | Path,
        one_root: str | Path,
    ):
        for label, value in (("modality", modality), ("subject", subject)):
            if not value or value in (".", "..") or any(c in value for c in "/\\:"):
                raise ValueError(f"{label} must be one directory name: {value!r}")

        try:
            if len(date) == 8 and date.isdigit():
                date = Date.fromisoformat(f"{date[:4]}-{date[4:6]}-{date[6:8]}").isoformat()
            elif Date.fromisoformat(date).isoformat() != date:
                raise ValueError
        except ValueError:
            raise ValueError(f"date must use YYYY-MM-DD or YYYYMMDD: {date!r}")

        if isinstance(session_number, bool) or not isinstance(session_number, int) or session_number < 1:
            raise ValueError(f"session_number must be a positive integer: {session_number!r}")

        self.modality = modality
        self.subject = subject
        self.date = date
        self.session_number = session_number

        self.raw_data_root = Path(raw_data_root)
        self.processed_data_root = Path(processed_data_root)
        # ONE's cache root: converted sessions are written directly below it as
        # subject/date/number, and ONE indexes them from here.  The modality
        # only appears in the raw and processed source paths.
        self.one_root = Path(one_root)

        self.relative_path = Path(subject) / date / f"{session_number:03d}"
        source_relative_path = Path(modality) / subject / date.replace("-", "") / str(session_number)
        self.raw_path = self.raw_data_root / source_relative_path
        self.session_time_zero = float((self.raw_path / "timestamped_digital.csv").open().readline().split(",", 1)[0])

        # This is the session output root, containing alf/ and raw_passive_data/.
        self.session_path = self.one_root / self.relative_path

        self.exporters = self.get_exporters()

    def get_exporters(self) -> Exporters:
        """Return the stable exporter instances available for this session."""
        if hasattr(self, "exporters"):
            return self.exporters

        # Import lazily to avoid a circular import: exporters use the Collection
        # and IBLDataset types defined above in this module.
        from .exporters import REGISTERED_EXPORTERS

        # Instantiate each registered exporter, then retain only exporters
        # whose get_collections() finds a valid acquisition for this session.
        # The retained instances remain stable so users can adjust settings.
        exporters = Exporters()
        for exporter_class in REGISTERED_EXPORTERS:
            exporter = exporter_class(self)
            if exporter.get_collections():
                print(repr(exporter))
                exporters.append(exporter)
        return exporters

    def export(self) -> list[Path]:
        """Export this session's registered datasets and return their paths.

        Export functions receive their owning exporter so they can use its
        session and any user-adjusted exporter settings.
        ``Collection.walk`` supplies paths relative to the session output
        root, including the collection directory and ALF filename.
        """
        exported_paths: list[Path] = []

        exporter_collections = [
            (exporter, exporter.get_collections()) for exporter in self.exporters
        ]
        self.collections = [
            collection
            for exporter, collections in exporter_collections
            for collection in collections
        ]

        for exporter, collections in exporter_collections:
            for collection in collections:
                for relative_output_path, dataset in collection.walk():
                    output_path = self.session_path / relative_output_path
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    exported_paths.append(dataset.export(exporter, output_path))

        return exported_paths



def explain_dataset(dataset: str | Path) -> None:
    """Print documentation for an ALF dataset filename or relative path."""
    from .exporters import REGISTERED_EXPORTERS

    filename = str(dataset).strip().replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    matches = [
        definition
        for exporter_class in REGISTERED_EXPORTERS
        for definition in exporter_class.datasets.values()
        if definition.filename == filename
    ]

    if not matches:
        raise KeyError(f"Unknown dataset {filename!r}")

    docstrings = {definition.docstring for definition in matches}
    if len(docstrings) > 1:
        exporter_names = ", ".join(
            exporter_class.__name__
            for exporter_class in REGISTERED_EXPORTERS
            if any(definition.filename == filename for definition in exporter_class.datasets.values())
        )
        raise ValueError(
            f"Ambiguous dataset {filename!r}: conflicting documentation in {exporter_names}"
        )

    print(matches[0].docstring or "No documentation available.")
