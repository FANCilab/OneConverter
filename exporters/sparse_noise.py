"""Export sparse-noise receptive-field mapping data to documented ALF datasets."""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np

from ..core import Collection, IBLDataset
from .base import Exporter

PASSIVE_RFM_TIMES_DOC = """Dataset: _ibl_passiveRFM.times.npy
Units: seconds relative to session start
Type: float64
Dimensions: [nFrames]
Format: npy
Description: Interpolated presentation time for every frame in the receptive-field mapping period.
"""

RF_MAP_STIM_RAW_DOC = """Dataset: _iblrig_RFMapStim.raw.bin
Units: color
Type: int8
Dimensions: [nFrames, nCols, nRows]
Format: bin
Description: Raw receptive-field mapping stimulus matrix.
The source bytes are preserved unchanged. They are int8 in C order with shape
[nFrames, nCols, nRows] (frame-major). Each value marks a square as gray
(background), black or white.
"""

SPARSE_NOISE_XY_DOC = """Dataset: _ibl_sparseNoise.xy.npy
Units: the public dataset-types spreadsheet leaves the unit unresolved; this
exporter follows the previous implementation and writes visual degrees
Type: float64
Dimensions: [nSparseNoise, 2]
Format: npy
Description: Screen x/y coordinates of each non-gray sparse-noise stimulus square.
"""

SPARSE_NOISE_TIMES_DOC = """Dataset: _ibl_sparseNoise.times.npy
Units: seconds relative to session start
Type: float64
Dimensions: [nSparseNoise]
Format: npy
Description: Presentation time of each non-gray sparse-noise stimulus square.
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


def load_sparse_noise_movie(exporter: SparseNoiseExporter) -> tuple[np.ndarray, np.ndarray]:
    """Return the decoded `[frame, row, col]` movie and its absolute frame times.

    `SparseNoise_Log.bin` is frame-major int8 in C order with shape
    `(n_frames, n_columns, n_rows)`. The exporter's `stimulus_values` setting
    gives the byte value of each color; they are decoded as gray -> 0,
    black -> -1 and white -> +1.
    """
    converter = exporter.session_converter
    frame_times = np.ravel(
        np.loadtxt(
            converter.raw_path / "times_grid.csv",
            delimiter=",",
            ndmin=1,
        )
    ).astype(np.float64, copy=False)
    # One frame time per stimulus frame; the times must be usable as a clock.
    if not np.all(np.isfinite(frame_times)):
        raise ValueError("times_grid.csv contains non-finite frame times")
    if not np.all(np.diff(frame_times) > 0):
        raise ValueError("times_grid.csv frame times are not strictly increasing")
    n_frames = frame_times.size

    # The grid size is fixed by the exporter setting; no shape is inferred.
    n_rows, n_columns = exporter.sparse_noise_shape
    raw = np.fromfile(converter.raw_path / "SparseNoise_Log.bin", dtype=np.int8)
    expected_size = n_frames * n_rows * n_columns
    if raw.size != expected_size:
        raise ValueError(
            f"SparseNoise_Log.bin has {raw.size} bytes but times_grid.csv has "
            f"{n_frames} frames; expected {n_frames} x {n_rows} x {n_columns} "
            f"= {expected_size} bytes"
        )
    # Reorder the stored [frame, col, row] layout to [frame, row, col].
    movie = raw.reshape(n_frames, n_columns, n_rows).transpose(0, 2, 1)

    # Only the byte values configured for gray, black and white may appear.
    stimulus_values = exporter.stimulus_values
    unexpected = np.setdiff1d(np.unique(movie), list(stimulus_values.values()))
    if unexpected.size:
        allowed = ", ".join(f"{value} ({color})" for color, value in stimulus_values.items())
        raise ValueError(
            f"SparseNoise_Log.bin contains unexpected values {unexpected.tolist()}; "
            f"only {allowed} are allowed"
        )
    # Decode black to -1 and white to +1; the remaining gray squares become 0.
    movie = np.select(
        (movie == stimulus_values["black"], movie == stimulus_values["white"]),
        (-1, 1),
        default=0,
    )
    return movie, frame_times


def export_sparse_noise_xy(exporter: SparseNoiseExporter, output_path: Path) -> Path:
    """Write `_ibl_sparseNoise.xy.npy` with one x/y location per event."""
    movie, _ = load_sparse_noise_movie(exporter)
    _, n_rows, n_columns = movie.shape

    # Row 0 is the bottom of the screen and column 0 is its left edge.
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
    movie, frame_times = load_sparse_noise_movie(exporter)

    # np.nonzero walks the movie in C order, matching the xy event order.
    frame_indices = np.nonzero(movie)[0]
    event_times = (frame_times - converter.session_time_zero)[frame_indices].astype(
        np.float64, copy=False
    )
    np.save(output_path, event_times)
    return output_path


class SparseNoiseExporter(Exporter):
    settings = {
        "sparse_noise_shape": (7, 27),
        # Byte value of each square color in SparseNoise_Log.bin.
        "stimulus_values": {"gray": -128, "black": -1, "white": 0},
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
