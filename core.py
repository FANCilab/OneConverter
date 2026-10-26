from __future__ import annotations

import json
import os
from datetime import date as Date
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Iterator

from one.alf import spec
from one.alf.cache import make_parquet_db
from one.alf.path import PureALFPath

# Dataset descriptions live in this file at the top of the ONE root, next to
# ONE's sessions.pqt and datasets.pqt.  ONE ignores it: it is outside every
# session folder and is not a .pqt table.
DATASET_TYPES_FILENAME = "dataset_types.json"


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

    def export(self, force_export: bool = False) -> list[Path]:
        """Export this session's registered datasets and return their paths.

        Export functions receive their owning exporter so they can use its
        session and any user-adjusted exporter settings.
        ``Collection.walk`` supplies paths relative to the session output
        root, including the collection directory and ALF filename.

        If the session output directory already contains files, nothing is
        written and an empty list is returned.  Pass ``force_export=True`` to
        export again and overwrite the existing datasets.

        Every call ends by rebuilding the ONE cache tables (``sessions.pqt``
        and ``datasets.pqt``) in ``one_root`` with
        ``one.alf.cache.make_parquet_db``, so ``ONE(cache_dir=one_root)``
        always indexes the current contents.  This happens even when the
        session was already exported.

        Every call also writes the documentation of all converter dataset
        types into ``one_root/dataset_types.json`` (see
        ``update_dataset_types``), where ``explain_dataset`` reads it.
        """
        exported_paths: list[Path] = []

        # A session counts as already exported when any file exists below its
        # output root; individual datasets are not checked.
        if not force_export and self.session_path.is_dir() and any(
            path.is_file() for path in self.session_path.rglob("*")
        ):
            print(
                f"{self.session_path} is already exported; "
                "use export(force_export=True) to export again."
            )
        else:
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

        # Re-index the whole ONE root on every call, including when the
        # session was already exported, so the parquet cache is never stale.
        make_parquet_db(self.one_root)

        # Import lazily to avoid a circular import, as in get_exporters().
        from .exporters import REGISTERED_EXPORTERS

        # Publish the documentation of every dataset type the converter knows,
        # not only this session's, so docstring edits reach the JSON file on
        # the next export() call of any session.
        update_dataset_types(self.one_root, {
            dataset.filename: dataset.docstring
            for exporter_class in REGISTERED_EXPORTERS
            for dataset in exporter_class.datasets.values()
        })

        return exported_paths


def update_dataset_types(one_root: str | Path, descriptions: dict[str, str | None]) -> Path:
    """Add or replace dataset descriptions in ``one_root/dataset_types.json``.

    ``descriptions`` maps ALF dataset filenames (namespace, object, attribute
    and extension, without the collection), such as
    ``"_ibl_wheel.position.npy"``, to their documentation.  Entries already in
    the file are kept; entries with the same filename are replaced, like
    ``dict.update``.  Both ``OneSessionConverter.export`` and scripts that
    write their own datasets below ``one_root`` use this function, so
    ``explain_dataset`` can document any of them.

    Returns the path of the JSON file.
    """
    dataset_types_path = Path(one_root) / DATASET_TYPES_FILENAME

    # Start from the existing entries so descriptions written by other
    # scripts survive this update.
    dataset_types = {}
    if dataset_types_path.is_file():
        dataset_types = json.loads(dataset_types_path.read_text(encoding="utf-8"))

    # Each entry is an object so fields such as units can be added later
    # without changing the file layout.
    for filename, description in descriptions.items():
        dataset_types[filename] = {"description": description}

    # Write a temporary file and swap it in, so an interrupted write never
    # leaves a truncated JSON file behind.
    temporary_path = dataset_types_path.with_name(DATASET_TYPES_FILENAME + ".tmp")
    temporary_path.write_text(
        json.dumps(dataset_types, indent=1, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary_path, dataset_types_path)
    return dataset_types_path


def explain_dataset(one: object, dataset: str | Path) -> None:
    """Print documentation for an ALF dataset filename or relative path.

    The documentation is read from ``dataset_types.json`` in the cache
    directory of ``one`` (the ``one_root`` the session was exported to), so
    any dataset registered with ``update_dataset_types`` can be explained.
    """
    # Keep only the filename: the JSON file is keyed without collections.
    filename = str(dataset).strip().replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]

    dataset_types_path = Path(one.cache_dir) / DATASET_TYPES_FILENAME
    dataset_types = json.loads(dataset_types_path.read_text(encoding="utf-8"))

    if filename not in dataset_types:
        raise KeyError(f"Unknown dataset {filename!r}")

    print(dataset_types[filename]["description"] or "No documentation available.")
