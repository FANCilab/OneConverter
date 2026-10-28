"""Export rotary encoder events as IBL wheel datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core import Collection, IBLDataset
from .base import Exporter


WHEEL_POSITION_DOC = """Dataset: _ibl_wheel.position.npy
Units: radians
Type: float64
Dimensions: [number of wheel ticks]
Format: npy
Description: Absolute unwrapped angle of the wheel from session start. The sign from the subject perspective corresponds to mathematical convention: counter clockwise is positive."""

WHEEL_TIMESTAMPS_DOC = """Dataset: _ibl_wheel.timestamps.npy
Units: seconds relative to session start
Type: float64
Dimensions: [number of wheel ticks]
Format: npy
Description: Time in seconds of the corresponding wheel event, relative to session start. Note that the wheel delivers a sample only when crossing a tick, which makes this time-serie very irregular."""

WHEEL_VELOCITY_DOC = "Description: Docstring placeholder for _ibl_wheel.velocity dataset"


def _wheel_events(exporter: WheelExporter) -> tuple[np.ndarray, np.ndarray]:
    """Select encoder count changes after session start and unwrap the 16 bit counter."""
    converter = exporter.session_converter
    samples = np.loadtxt(converter.raw_path / "encoder_log.csv", delimiter=",", ndmin=2)
    samples = samples[samples[:, 0] >= converter.session_time_zero]
    timestamps = samples[:, 0].astype(np.float64, copy=False)
    counts = samples[:, 1]

    # The acquisition logs the signed 16 bit counter every millisecond. IBL
    # wheel rows represent tick events, so repeated counts are omitted.
    count_steps = np.diff(counts)
    count_steps[count_steps > 32768] -= 65536
    count_steps[count_steps < -32768] += 65536
    unwrapped = np.concatenate(([0.0], np.cumsum(count_steps)))
    tick_rows = np.r_[False, count_steps != 0]
    position = unwrapped[tick_rows] * (2 * np.pi / exporter.encoder_counts_per_revolution)
    return timestamps[tick_rows] - converter.session_time_zero, position


def export_wheel_position(exporter: WheelExporter, output_path: Path) -> Path:
    """Write unwrapped wheel angle at each encoder tick."""
    _, position = _wheel_events(exporter)
    np.save(output_path, position)
    return output_path


def export_wheel_timestamps(exporter: WheelExporter, output_path: Path) -> Path:
    """Write session relative time at each encoder tick."""
    timestamps, _ = _wheel_events(exporter)
    np.save(output_path, timestamps)
    return output_path


def export_wheel_velocity(exporter: WheelExporter, output_path: Path) -> Path:
    """Write the change in wheel angle per second between successive ticks."""
    timestamps, position = _wheel_events(exporter)
    velocity = np.zeros_like(position)
    velocity[1:] = np.diff(position) / np.diff(timestamps)
    np.save(output_path, velocity)
    return output_path


class WheelExporter(Exporter):
    # The IBL dataset-types document describes a 1024-resolution encoder
    # with X4 decoding: 4096 ticks per revolution. This is independent of
    # the 16 bit counter wrap handled in _wheel_events().
    settings = {"encoder_counts_per_revolution": 4096}
    datasets = {
        "_ibl_wheel.position.npy": IBLDataset(
            ibl_object="wheel",
            attribute="position",
            extension="npy",
            namespace="ibl",
            export_fn=export_wheel_position,
            docstring=WHEEL_POSITION_DOC,
        ),
        "_ibl_wheel.timestamps.npy": IBLDataset(
            ibl_object="wheel",
            attribute="timestamps",
            extension="npy",
            namespace="ibl",
            export_fn=export_wheel_timestamps,
            docstring=WHEEL_TIMESTAMPS_DOC,
        ),
        "_ibl_wheel.velocity.npy": IBLDataset(
            ibl_object="wheel",
            attribute="velocity",
            extension="npy",
            namespace="ibl",
            export_fn=export_wheel_velocity,
            docstring=WHEEL_VELOCITY_DOC,
        ),
    }

    def get_collections(self) -> list[Collection]:
        """Expose wheel data when the raw session contains encoder samples."""
        if not (self.session_converter.raw_path / "encoder_log.csv").is_file():
            return []
        return [Collection("alf", datasets=list(self.datasets.values()))]
