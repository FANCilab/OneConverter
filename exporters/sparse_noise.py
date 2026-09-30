"""Export sparse-noise receptive-field mapping data to documented ALF datasets."""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np

from ..core import Collection, IBLDataset
from .base import Exporter

PASSIVE_RFM_TIMES_DOC = """Interpolated presentation time for every frame in the receptive-field mapping period.

Dataset: _ibl_passiveRFM.times.npy
Units: seconds relative to session start
Type: float64
Dimensions: [nFrames]
Format: npy
"""

RF_MAP_STIM_RAW_DOC = """Raw receptive-field mapping stimulus matrix.

Dataset: _iblrig_RFMapStim.raw.bin
Units: color
Type: uint8
Dimensions: [nFrames, nx, ny]
Format: bin

The public IBL dataset-types document specifies nx == ny == 15. Legacy source
sessions may instead use 10 x 30 or 7 x 27 layouts; the source bytes are
preserved in the exported dataset.
"""

SPARSE_NOISE_XY_DOC = """Screen x/y coordinates of each non-gray sparse-noise stimulus square.

Dataset: _ibl_sparseNoise.xy.npy
Units: the public dataset-types spreadsheet leaves the unit unresolved; this
exporter follows the previous implementation and writes visual degrees
Type: float64
Dimensions: [nSparseNoise, 2]
Format: npy
"""

SPARSE_NOISE_TIMES_DOC = """Presentation time of each non-gray sparse-noise stimulus square.

Dataset: _ibl_sparseNoise.times.npy
Units: seconds relative to session start
Type: float64
Dimensions: [nSparseNoise]
Format: npy
"""


def export_passive_rfm_times(exporter: SparseNoiseExporter, output_path: Path) -> Path:
    """Write `_ibl_passiveRFM.times.npy` as session-relative float64 seconds."""
    converter = exporter.session_converter
    frame_times = np.ravel(
        np.loadtxt(
            converter.raw_path / "times_grid.csv",
            delimiter=",",
            ndmin=1,
        )
    ).astype(np.float64, copy=False)
    np.save(output_path, frame_times - converter.session_time_zero)
    return output_path


def export_rf_map_stim_raw(exporter: SparseNoiseExporter, output_path: Path) -> Path:
    """Write `_iblrig_RFMapStim.raw.bin`, preserving the source bytes."""
    converter = exporter.session_converter
    shutil.copyfile(converter.raw_path / "SparseNoise_Log.bin", output_path)
    return output_path


def export_sparse_noise_xy(exporter: SparseNoiseExporter, output_path: Path) -> Path:
    """Write `_ibl_sparseNoise.xy.npy` with one x/y location per event."""
    converter = exporter.session_converter
    raw = np.fromfile(converter.raw_path / "SparseNoise_Log.bin", dtype=np.int8)
    shape = next(
        (
            (rows, columns)
            for rows, columns in exporter.sparse_noise_shapes
            if raw.size % (rows * columns) == 0
        ),
    )
    n_rows, n_columns = shape
    movie = np.moveaxis(raw.reshape(n_rows, n_columns, -1), 2, 0)
    movie = np.select(
        (movie == -1, movie == 0),
        (-1, 1),
        default=0,
    )

    left, right, bottom, top = exporter.stimulus_edges_deg
    x_edges = np.linspace(left, right, n_columns + 1)
    y_edges = np.linspace(bottom, top, n_rows + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    _, row_indices, column_indices = np.nonzero(movie)
    positions = np.column_stack((x_centers[column_indices], y_centers[row_indices])).astype(
        np.float64, copy=False
    )
    np.save(output_path, positions)
    return output_path


def export_sparse_noise_times(exporter: SparseNoiseExporter, output_path: Path) -> Path:
    """Write `_ibl_sparseNoise.times.npy` with one session-relative time per event."""
    converter = exporter.session_converter
    raw = np.fromfile(converter.raw_path / "SparseNoise_Log.bin", dtype=np.int8)
    shape = next(
        (
            (rows, columns)
            for rows, columns in exporter.sparse_noise_shapes
            if raw.size % (rows * columns) == 0
        ),
    )
    n_rows, n_columns = shape
    movie = np.moveaxis(raw.reshape(n_rows, n_columns, -1), 2, 0)
    movie = np.select(
        (movie == -1, movie == 0),
        (-1, 1),
        default=0,
    )
    frame_times = np.ravel(
        np.loadtxt(
            converter.raw_path / "times_grid.csv",
            delimiter=",",
            ndmin=1,
        )
    ).astype(np.float64, copy=False)

    frame_indices = np.nonzero(movie)[0]
    event_times = (frame_times - converter.session_time_zero)[frame_indices].astype(
        np.float64, copy=False
    )
    np.save(output_path, event_times)
    return output_path


class SparseNoiseExporter(Exporter):
    settings = {
        "sparse_noise_shapes": ((15, 15), (10, 30), (7, 27)),
        "stimulus_edges_deg": (-135.0, 135.0, -40.0, 40.0),
    }
    datasets = {
        "_ibl_passiveRFM.times.npy": IBLDataset(
            ibl_object="passiveRFM",
            attribute="times",
            extension="npy",
            namespace="ibl",
            export_fn=export_passive_rfm_times,
            docstring=PASSIVE_RFM_TIMES_DOC,
        ),
        "_ibl_sparseNoise.times.npy": IBLDataset(
            ibl_object="sparseNoise",
            attribute="times",
            extension="npy",
            namespace="ibl",
            export_fn=export_sparse_noise_times,
            docstring=SPARSE_NOISE_TIMES_DOC,
        ),
        "_ibl_sparseNoise.xy.npy": IBLDataset(
            ibl_object="sparseNoise",
            attribute="xy",
            extension="npy",
            namespace="ibl",
            export_fn=export_sparse_noise_xy,
            docstring=SPARSE_NOISE_XY_DOC,
        ),
        "_iblrig_RFMapStim.raw.bin": IBLDataset(
            ibl_object="RF_Map_Stim",
            attribute="raw",
            extension="bin",
            namespace="iblrig",
            export_fn=export_rf_map_stim_raw,
            docstring=RF_MAP_STIM_RAW_DOC,
        ),
    }

    def get_collections(self) -> list[Collection]:
        """Return the documented ALF and raw-passive sparse-noise collections."""
        # A complete sparse-noise acquisition is required before this exporter
        # exposes its collections.
        required_raw_files = (
            "SparseNoise_Log.bin",
            "times_grid.csv",
            "timestamped_digital.csv",
        )
        if not all(
            (self.session_converter.raw_path / filename).is_file()
            for filename in required_raw_files
        ):
            return []

        alf = Collection(
            "alf",
            datasets=[
                self.datasets["_ibl_passiveRFM.times.npy"],
                self.datasets["_ibl_sparseNoise.times.npy"],
                self.datasets["_ibl_sparseNoise.xy.npy"],
            ],
        )
        raw_passive_data = Collection(
            "raw_passive_data",
            datasets=[self.datasets["_iblrig_RFMapStim.raw.bin"]],
        )
        return [alf, raw_passive_data]
