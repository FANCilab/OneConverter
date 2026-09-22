from __future__ import annotations

import csv
import json
import os
import pathlib
import re
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any

import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None


FILE_SPEC = r"_?{namespace}?_?{object}\.{attribute}(?:_{timescale})?(?:\.{extra})*\.{extension}$"

_DEFAULT = (
    ("namespace", "(?<=_)[a-zA-Z0-9]+"),
    ("object", r"\w+"),
    ("attribute", r"(?:_[a-z]+_)?[a-zA-Z0-9]+(?:_times(?=[_.])|_intervals(?=[_.]))?"),
    ("timescale", r"\w+"),
    ("extra", r"[.\w-]+"),
    ("extension", r"\w+"),
)


class ALFInvalid(ValueError):
    pass


def _dromedary(string: str) -> str:
    def capitalize(part: str) -> str:
        return part if re.match(r"^[A-Z]+s?$", part) else part.capitalize()

    if not string:
        return string
    first, *other = re.split(r"[_\s]", string)
    if not other:
        return first[0].lower() + first[1:]
    first = first if re.match(r"^[A-Z]+s?$", first) else first.lower()
    return "".join((first, *map(capitalize, other)))


def _named(pattern: str, name: str) -> str:
    return f"(?P<{name}>{pattern})"


def _regex(spec_str: str = FILE_SPEC, **kwargs: str | None) -> re.Pattern:
    fields = dict(_DEFAULT)
    if not fields.keys() >= kwargs.keys():
        unknown = next(k for k in kwargs.keys() if k not in fields.keys())
        raise KeyError(f'Unknown field "{unknown}"')
    fields.update({k: v for k, v in kwargs.items() if v is not None})
    return re.compile(spec_str.format(**{k: _named(fields[k], k) for k in re.findall(r"(?<={)\w+", spec_str)}))


def _is_valid(filename: str) -> bool:
    return _regex(FILE_SPEC).match(filename) is not None


def _to_alf(
    object: str,
    attribute: str,
    extension: str,
    namespace: str | None = None,
    timescale: str | tuple[str, ...] | None = None,
    extra: str | tuple[str, ...] | None = None,
) -> str:
    if not extension:
        raise TypeError("An extension must be provided")
    if extension.startswith("."):
        extension = extension[1:]
    if any(part is not None and "." in part for part in (object, attribute, namespace, extension, timescale)):
        raise ValueError("ALF parts must not contain a period (`.`)")
    if "_" in (namespace or ""):
        raise ValueError("Namespace must not contain extra underscores")
    if object[0] == "_":
        raise ValueError("Objects must not contain underscores; use namespace arg instead")

    if timescale:
        timescale_parts = [timescale] if isinstance(timescale, str) else timescale
        timescale = "_".join(map(_dromedary, filter(None, timescale_parts)))

    times_match = re.search("_(times|timestamps|intervals)$", attribute)
    split_at = times_match.start() if times_match else len(attribute)
    attribute = _dromedary(attribute[:split_at]) + attribute[split_at:]
    object = _dromedary(object)

    if not extra:
        extra = ()
    elif isinstance(extra, str):
        extra = tuple(extra.split("."))

    parts = (
        (f"_{namespace}_" if namespace else "") + object,
        attribute + (f"_{timescale}" if timescale else ""),
        *extra,
        extension,
    )
    return ".".join(parts)


spec = SimpleNamespace(to_alf=_to_alf, is_valid=_is_valid, regex=_regex, FILE_SPEC=FILE_SPEC)


def _path_parts(path: str | pathlib.PurePath, spec_str: str, as_dict: bool = False, assert_valid: bool = True):
    if hasattr(path, "as_posix"):
        path = path.as_posix()
    pattern = _regex(spec_str)
    empty = OrderedDict.fromkeys(pattern.groupindex.keys())
    parsed = pattern.match(path)
    if parsed:
        parsed_dict = parsed.groupdict()
        return OrderedDict(parsed_dict) if as_dict else tuple(parsed_dict.values())
    if assert_valid:
        raise ALFInvalid(path)
    return empty if as_dict else tuple(empty.values())


def _filename_parts(filename: str, as_dict: bool = False, assert_valid: bool = True):
    return _path_parts(filename, FILE_SPEC, as_dict, assert_valid)


class PureALFPath(pathlib.PurePath):
    def __new__(cls, *args: str | pathlib.PurePath):
        if cls is PureALFPath:
            cls = PureWindowsALFPath if os.name == "nt" else PurePosixALFPath
        return super().__new__(cls, *args)

    def is_dataset(self) -> bool:
        return _is_valid(self.name)

    def parse_alf_name(self, as_dict: bool = True):
        return _filename_parts(self.name, as_dict=as_dict, assert_valid=False)


class PureWindowsALFPath(pathlib.PureWindowsPath, PureALFPath):
    pass


class PurePosixALFPath(pathlib.PurePosixPath, PureALFPath):
    pass


def _ensure_flat(array: np.ndarray) -> np.ndarray:
    return array.flatten() if array.ndim == 2 and array.shape[1] == 1 else array


def _read_table(path: pathlib.Path, delimiter: str) -> list[list[str]]:
    with path.open(newline="") as file:
        return list(csv.reader(file, delimiter=delimiter))


def load_file_content(fil: str | os.PathLike[str]) -> Any:
    if not fil:
        return None

    path = pathlib.Path(fil)
    if path.stat().st_size == 0:
        return None

    if path.suffix == ".json":
        with path.open() as file:
            return json.load(file)
    if path.suffix == ".npy":
        return _ensure_flat(np.load(file=path, allow_pickle=True))
    if path.suffix == ".npz":
        array = np.load(file=path)
        return array["arr_0"] if set(array.files) == {"arr_0"} else array
    if path.suffix == ".csv":
        if pd is not None:
            return pd.read_csv(path).squeeze("columns")
        return _read_table(path, ",")
    if path.suffix == ".ssv":
        if pd is not None:
            return pd.read_csv(path, delimiter=" ").squeeze("columns")
        return _read_table(path, " ")
    if path.suffix in (".tsv", ".htsv"):
        if pd is not None:
            return pd.read_csv(path, delimiter="\t").squeeze("columns")
        return _read_table(path, "\t")
    if path.suffix in (".yml", ".yaml"):
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(f"{path.name} requires PyYAML to load.") from exc
        with path.open() as file:
            return yaml.safe_load(file)

    return path
