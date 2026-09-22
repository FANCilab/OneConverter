from .one import PureALFPath, spec
from datetime import date as Date
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Iterator
RAW_DATA_PATH = Path("//10.233.25.135/FANCiNAS1/Data") #READ ONLY
PROCESSED_DATA_PATH = Path("//10.233.25.135/FANCiNAS1/Processed") #READ ONLY
ALF_ROOT = Path("D:/alftoolbox")


@dataclass
class IBLDataset:
    IBLobject: str
    attribute: str
    extension: str
    export_fn: Callable[[dict[str, Path], Path], Path]
    namespace: str | None = None
    timescale: str | None = None
    docstring: str | None = None
    extra: tuple[str, ...] = ()
    _path: PureALFPath = field(init=False, repr=False)

    def __post_init__(self) -> None:
        filename = spec.to_alf(
            self.IBLobject, self.attribute, self.extension,
            namespace=self.namespace, timescale=self.timescale, extra=self.extra,
        )
        self._path = PureALFPath(filename)
        if not self._path.is_dataset():
            raise ValueError(f"Invalid ALF dataset filename: {filename}")

    @property
    def filename(self) -> str:
        return self._path.name

    def export(self,source_paths:list[Path], output_path: Path) -> Path:
        return self.export_fn(source_paths, output_path)


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



class OneSession:
    def __init__(self, modality: str, subject: str, date: str, session_number: int):
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

        self.relative_path = Path(modality) / subject / date / str(session_number)
        self.raw_path = RAW_DATA_PATH / self.relative_path
        self.processed_path = PROCESSED_DATA_PATH / self.relative_path

        # This is the session output root, containing alf/ and raw_passive_data/.
        self.alf_path = ALF_ROOT / self.relative_path

        # Import lazily to avoid a circular import: exporters use the Collection
        # and IBLDataset types defined above in this module.
        from .exporters import REGISTERED_EXPORTERS

        self.collections = [
            collection
            for exporter in REGISTERED_EXPORTERS
            for collection in exporter.get_collections(self)
        ]
