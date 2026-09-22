"""Export sparse-noise RF mapping stimuli to ALF/ONE datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .calcium_export import _session_time_zero, _to_session_relative_times
    from .wheel_export import _find_raw_file, _session_from_run_db
except ImportError:
    from calcium_export import _session_time_zero, _to_session_relative_times
    from wheel_export import _find_raw_file, _session_from_run_db


DEFAULT_STIMULUS_EDGES_DEG = (-135.0, 135.0, -40.0, 40.0)
SPARSE_NOISE_SHAPES = ((10, 30), (7, 27))
SPARSE_NOISE_VALUES = {-1, 0, 1}


def single_session_sparse_noise_export(
    s2prunpath,
    dataoutroot,
    *,
    stimulus_edges_deg=DEFAULT_STIMULUS_EDGES_DEG,
):
    """Export one session's sparse-noise stimulus data to ALF datasets."""

    s2p_run_path = Path(s2prunpath)
    data_out_root = Path(dataoutroot)
    subject, date, session, raw_path = _session_from_run_db(s2p_run_path)

    sparse_noise_path = _find_raw_file(raw_path, "SparseNoise_Log.bin")
    times_grid_path = _find_raw_file(raw_path, "times_grid.csv")

    stim_map = _load_sparse_noise_movie(sparse_noise_path)
    frame_times = _load_frame_times(times_grid_path, raw_path)
    stim_map, frame_times = _align_movie_and_times(stim_map, frame_times)
    sparse_times, sparse_positions = _sparse_noise_events(
        stim_map,
        frame_times,
        stimulus_edges_deg,
    )

    alf_folder = data_out_root / subject / date / session / "alf"
    raw_passive_folder = data_out_root / subject / date / session / "raw_passive_data"

    alf_folder.mkdir(parents=True, exist_ok=True)
    raw_passive_folder.mkdir(parents=True, exist_ok=True)

    written = [
        _save_npy(alf_folder / "_ibl_passiveRFM.times.npy", frame_times.astype(np.float64, copy=False)),
        _save_sparse_noise_raw_bin(
            raw_passive_folder / "_iblrig_RFMapStim.raw.bin",
            sparse_noise_path,
            expected_shape=stim_map.shape,
        ),
        _save_npy(alf_folder / "_ibl_sparseNoise.times.npy", sparse_times.astype(np.float64, copy=False)),
        _save_npy(alf_folder / "_ibl_sparseNoise.xy.npy", sparse_positions.astype(np.float64, copy=False)),
    ]
    return written


def _load_sparse_noise_movie(path: Path) -> np.ndarray:
    """Load and decode SparseNoise_Log.bin without dropping frames.

    The exported movie is an int8 array with values -1, 0, and 1, where
    -1 = black/OFF, 0 = gray/no stimulus, and 1 = white/ON.
    """

    raw = np.fromfile(path, dtype=np.int8)
    if raw.size == 0:
        raise ValueError(f"Sparse-noise movie is empty: {path}")

    for n_rows, n_cols in SPARSE_NOISE_SHAPES:
        pixels_per_frame = n_rows * n_cols
        if raw.size % pixels_per_frame == 0:
            movie = raw.reshape((n_rows, n_cols, raw.size // pixels_per_frame)).copy()
            movie = _decode_sparse_noise_values(movie, path)
            return np.moveaxis(movie, 2, 0).astype(np.int8, copy=False)

    raise ValueError(
        f"Could not infer sparse-noise frame shape for {path}; "
        f"{raw.size} int8 values is not divisible by {[r * c for r, c in SPARSE_NOISE_SHAPES]}"
    )


def _decode_sparse_noise_values(movie: np.ndarray, path: Path) -> np.ndarray:
    """Decode raw SparseNoise_Log.bin int8 values to RF-analysis values.

    Output convention:
        -1 = black / OFF
         0 = gray / blank
        +1 = white / ON

    Known local Bonsai/MATLAB encoding:
        raw -128 -> decoded 0
        raw -1   -> decoded -1
        raw 0    -> decoded +1
    """

    values = set(np.unique(movie).astype(int).tolist())

    # Already decoded signed sparse-noise convention.
    if values.issubset({-1, 0, 1}) and -128 not in values:
        return movie.astype(np.int8, copy=False)

    # Your local raw SparseNoise_Log.bin encoding.
    # MATLAB old loader:
    #   stimulus(stimulus == 0) = +1;
    #   stimulus(stimulus == -128) = 0;
    # leaving -1 unchanged.
    if values.issubset({-128, -1, 0}):
        decoded = np.empty_like(movie, dtype=np.int8)
        decoded[movie == -128] = 0
        decoded[movie == -1] = -1
        decoded[movie == 0] = 1
        return decoded

    # Common unsigned-ish int8 sparse-noise encoding.
    if values.issubset({-128, 0, 127}):
        decoded = np.empty_like(movie, dtype=np.int8)
        decoded[movie == -128] = -1
        decoded[movie == 0] = 0
        decoded[movie == 127] = 1
        return decoded

    raise ValueError(
        f"Unexpected SparseNoise_Log.bin values in {path}: {sorted(values)}. "
        "Expected known sparse-noise encoding decodable to -1/0/+1."
    )


def _load_frame_times(times_grid_path: Path, raw_path: Path) -> np.ndarray:
    times = np.loadtxt(times_grid_path, delimiter=",", ndmin=1).astype(np.float64, copy=False)
    times = np.ravel(times)
    return _to_session_relative_times(
        times,
        _session_time_zero(raw_path),
        description=f"sparse-noise frame times from {times_grid_path}",
    )


def _align_movie_and_times(stim_map: np.ndarray, frame_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate sparse-noise movie/timing alignment without trimming."""

    frame_times = np.asarray(frame_times, dtype=np.float64).reshape(-1)
    if stim_map.ndim != 3:
        raise ValueError(f"Sparse-noise movie must be 3D [nFrames, nRows, nCols], got {stim_map.shape}")
    if stim_map.shape[0] == 0 or frame_times.size == 0:
        raise ValueError("Sparse-noise movie and frame times must both be non-empty")
    if stim_map.shape[0] != frame_times.size:
        raise ValueError(
            "Sparse-noise movie and frame-time counts differ; refusing to trim data: "
            f"movie has {stim_map.shape[0]} frames, times has {frame_times.size} samples"
        )
    if not np.all(np.isfinite(frame_times)):
        raise ValueError("Sparse-noise frame times contain non-finite values")
    if frame_times.size > 1 and np.any(np.diff(frame_times) <= 0):
        raise ValueError("Sparse-noise frame times must be strictly increasing")
    return stim_map, frame_times


def _sparse_noise_events(
    stim_map: np.ndarray,
    frame_times: np.ndarray,
    stimulus_edges_deg,
) -> tuple[np.ndarray, np.ndarray]:
    if len(stimulus_edges_deg) != 4:
        raise ValueError("stimulus_edges_deg must be (left, right, bottom, top)")

    left, right, bottom, top = map(float, stimulus_edges_deg)
    n_frames, n_rows, n_cols = stim_map.shape
    x_edges = np.linspace(left, right, n_cols + 1)
    y_edges = np.linspace(bottom, top, n_rows + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    frame_indices, row_indices, col_indices = np.nonzero(stim_map != 0)
    sparse_times = frame_times[frame_indices]
    sparse_positions = np.column_stack(
        [
            x_centers[col_indices],
            y_centers[row_indices],
        ]
    )

    return sparse_times, sparse_positions


def _save_sparse_noise_raw_bin(
    path: Path,
    source_path: Path,
    *,
    expected_shape: tuple[int, int, int],
) -> Path:
    """Save registered IBL raw sparse-noise movie as uint8 .bin.

    This preserves the original SparseNoise_Log.bin bytes. The decoded
    stim_map is still used for _ibl_sparseNoise.times/xy event extraction,
    but the registered _iblrig_RFMapStim.raw.bin is raw uint8 data in the
    raw_passive_data collection.
    """

    raw = np.fromfile(source_path, dtype=np.uint8)
    expected_size = int(np.prod(expected_shape))

    if raw.size != expected_size:
        raise ValueError(
            f"Sparse-noise raw movie size does not match decoded movie shape: "
            f"{source_path} has {raw.size} bytes, expected {expected_size} "
            f"from shape {expected_shape}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    raw.tofile(path)
    return path


def _save_npy(path: Path, value: np.ndarray) -> Path:
    np.save(path, value)
    return path
