"""Suite2p multiphoton exporter."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import sparse

from ..core import Collection, IBLDataset
from .base import Exporter


MPCI_TIMES_DOC = """The time of each frame. Note that there may be further offsets available for each scanline or pixel/voxel in the mpciStack.timeshift.npy dataset.

Dataset: mpci.times.npy
Units: seconds relative to session start
Type: float64
Dimensions: [nframes]
Format: npy"""

MPCI_BAD_FRAMES_DOC = """Bad frames as identified by processing software. Typically these are frames where the image registration was considered poor.

Dataset: mpci.badFrames.npy
Units: N/A
Type: boolean
Dimensions: [nframes]
Format: npy"""

MPCI_ROI_ACTIVITY_F_DOC = """Mean activity of all pixels in each ROI.

Dataset: mpci.ROIActivityF.npy
Units: photodetector output units
Type: float32
Dimensions: [nFrames, nROIs]
Format: npy"""

MPCI_ROI_NEUROPIL_ACTIVITY_F_DOC = """Mean activity neuropil pixels neighboring each ROI.

Dataset: mpci.ROINeuropilActivityF.npy
Units: photodetector output units
Type: float32
Dimensions: [nFrames, nROIs]
Format: npy"""

MPCI_ROI_ACTIVITY_DECONVOLVED_DOC = """Neuropil-subtracted deconvolved activity of each ROI using standard parameters, which can be recovered from _suite2p_ROIData.raw.

Dataset: mpci.ROIActivityDeconvolved.npy
Units: AU
Type: float32
Dimensions: [nFrames, nROIs]
Format: npy"""

MPCI_ROI_TYPES_DOC = """Numerical enumeration of ROI type for each ROI.

Dataset: mpciROIs.mpciROITypes.npy
Units: N/A
Type: int16
Dimensions: [nROIs]
Format: npy

Values are the Suite2p manual curation label (iscell[:, 0]): 0 = nonCell,
1 = neuron."""

MPCI_CELL_CLASSIFIER_DOC = """Floating-point cell classifier score for each ROI, ranging between 0 and 1.

Dataset: mpciROIs.cellClassifier.npy
Units: N/A
Type: float64
Dimensions: [nROIs]
Format: npy"""

MPCI_ROI_MASKS_DOC = """Floating-point mask of each ROI, in 2D or 3D according to how you did detection. Saved as a sparse array. sum(roi_mask * frames[ypix,xpix,:]) = fluorescence

Dataset: mpciROIs.masks.sparse_npz
Units: AU
Type: float
Dimensions: [nROIs, H, W]
Format: sparse_npz"""

MPCI_STACK_POS_DOC = """Y, X, and Z (plane) pixel coordinates of each ROI's centroid.

Dataset: mpciROIs.stackPos.npy
Units: pixels
Type: int64
Dimensions: [nROIs, 3]
Format: npy"""

MPCI_MEAN_IMAGE_DOC = "Docstring placeholder for mpciMeanImage.images dataset"


def _fov_index(output_path: Path) -> int:
    """Read the FOV number from its ALF collection directory."""
    return int(output_path.parent.name.split("_")[-1])


def _export_frame_trace(exporter: MpciExporter, output_path: Path, source_name: str) -> Path:
    """Export a Suite2p ROI-by-frame trace as documented frame-by-ROI float32."""
    fov = exporter.fovs[_fov_index(output_path)]
    trace = np.load(fov["path"] / source_name, mmap_mode="r")
    np.save(output_path, trace[:, exporter.frame_start:exporter.frame_stop].T.astype(np.float32))
    return output_path


def _load_fov_iscell(exporter: MpciExporter, output_path: Path) -> np.ndarray:
    """Select combined curation rows belonging to this Suite2p plane."""
    fov = exporter.fovs[_fov_index(output_path)]
    iscell = np.load(fov["iscell_path"])
    if fov["iscell_path"].parent.name == "combined":
        combined_stat = np.load(fov["stat_path"], allow_pickle=True)
        plane = fov["plane_index"]
        iscell = iscell[
            np.fromiter((entry["iplane"] == plane for entry in combined_stat), dtype=bool)
        ]

        # The selected combined rows must cover exactly the ROIs traced in
        # this plane's F.npy (shape [nROIs, nFrames]).
        n_plane_rois = np.load(fov["path"] / "F.npy", mmap_mode="r").shape[0]
        if iscell.shape[0] != n_plane_rois:
            raise ValueError(
                f"combined/stat.npy has {iscell.shape[0]} rows for plane {plane}, but "
                f"{fov['path'] / 'F.npy'} has {n_plane_rois} ROIs"
            )

        # A plane-level iscell.npy holds the same classifier probabilities in
        # plane ROI order, so a mismatch means the combined rows are misordered.
        plane_iscell_path = fov["path"] / "iscell.npy"
        if plane_iscell_path.is_file():
            plane_iscell = np.load(plane_iscell_path)
            if plane_iscell.shape[0] != iscell.shape[0] or not np.allclose(
                plane_iscell[:, 1], iscell[:, 1], rtol=0, atol=1e-12
            ):
                raise ValueError(
                    f"Classifier probabilities in {plane_iscell_path} do not match the "
                    f"combined/iscell.npy rows selected for plane {plane}; the combined "
                    "row order does not match the plane ROI order"
                )
    return iscell


def export_mpci_times(exporter: MpciExporter, output_path: Path) -> Path:
    """Write one session-relative frame timestamp per FOV frame."""
    fov_index = _fov_index(output_path)
    np.save(output_path, exporter.times_by_fov[fov_index])
    return output_path


def export_mpci_roi_activity_f(exporter: MpciExporter, output_path: Path) -> Path:
    """Write Suite2p F as frame-by-ROI fluorescence."""
    return _export_frame_trace(exporter, output_path, "F.npy")


def export_mpci_bad_frames(exporter: MpciExporter, output_path: Path) -> Path:
    """Combine Suite2p registration flags with timing-derived bad frames."""
    fov = exporter.fovs[_fov_index(output_path)]
    ops = np.load(fov["path"] / "ops.npy", allow_pickle=True).item()
    bad_frames = np.asarray(ops.get("badframes", np.zeros(exporter.total_frame_count)), dtype=bool)
    bad_frames = bad_frames[exporter.frame_start:exporter.frame_stop]
    bad_frames |= exporter.timing_bad_frames_by_fov[_fov_index(output_path)]
    np.save(output_path, bad_frames)
    return output_path


def export_mpci_roi_neuropil_activity_f(exporter: MpciExporter, output_path: Path) -> Path:
    """Write Suite2p Fneu as frame-by-ROI neuropil fluorescence."""
    return _export_frame_trace(exporter, output_path, "Fneu.npy")


def export_mpci_roi_activity_deconvolved(exporter: MpciExporter, output_path: Path) -> Path:
    """Write Suite2p spks as frame-by-ROI deconvolved activity."""
    return _export_frame_trace(exporter, output_path, "spks.npy")


def export_mpci_roi_types(exporter: MpciExporter, output_path: Path) -> Path:
    """Write the Suite2p manual curation label (0 = nonCell, 1 = neuron) for each ROI."""
    np.save(output_path, _load_fov_iscell(exporter, output_path)[:, 0].astype(np.int16))
    return output_path


def export_mpci_cell_classifier(exporter: MpciExporter, output_path: Path) -> Path:
    """Write the Suite2p cell-classification probability for each ROI."""
    np.save(output_path, _load_fov_iscell(exporter, output_path)[:, 1].astype(np.float64))
    return output_path


def export_mpci_roi_masks(exporter: MpciExporter, output_path: Path) -> Path:
    """Write sparse weighted ROI masks from Suite2p stat pixel coordinates."""
    fov = exporter.fovs[_fov_index(output_path)]
    stat = np.load(fov["path"] / "stat.npy", allow_pickle=True)
    ops = np.load(fov["path"] / "ops.npy", allow_pickle=True).item()
    coordinates = []
    weights = []
    for roi_index, roi in enumerate(stat):
        ypix = np.asarray(roi["ypix"], dtype=np.int64)
        xpix = np.asarray(roi["xpix"], dtype=np.int64)
        coordinates.append(np.vstack((np.full(ypix.size, roi_index), ypix, xpix)))
        weights.append(np.asarray(roi["lam"], dtype=np.float32))
    coords = np.concatenate(coordinates, axis=1) if coordinates else np.empty((3, 0), dtype=np.int64)
    data = np.concatenate(weights) if weights else np.empty(0, dtype=np.float32)
    masks = sparse.COO(coords, data, shape=(len(stat), int(ops["Ly"]), int(ops["Lx"])))
    # sparse.save_npz appends .npz to path strings, so pass a stream to keep
    # the ALF extension supplied by the dataset definition.
    with output_path.open("wb") as stream:
        sparse.save_npz(stream, masks)
    return output_path


def export_mpci_stack_pos(exporter: MpciExporter, output_path: Path) -> Path:
    """Write each ROI centroid as Y, X, and plane pixel coordinates."""
    fov = exporter.fovs[_fov_index(output_path)]
    stat = np.load(fov["path"] / "stat.npy", allow_pickle=True)
    positions = np.empty((len(stat), 3), dtype=np.int64)
    for roi_index, roi in enumerate(stat):
        y, x = np.asarray(roi["med"])[:2]
        positions[roi_index] = (round(float(y)), round(float(x)), fov["plane_index"])
    np.save(output_path, positions)
    return output_path


def export_mpci_mean_image(exporter: MpciExporter, output_path: Path) -> Path:
    """Write the calcium-channel mean image with plane and channel axes."""
    fov = exporter.fovs[_fov_index(output_path)]
    ops = np.load(fov["path"] / "ops.npy", allow_pickle=True).item()
    image = np.asarray(ops["meanImg"], dtype=np.float64)
    np.save(output_path, image[:, :, np.newaxis, np.newaxis])
    return output_path


class MpciExporter(Exporter):
    settings = {
        # batch_list is populated lazily by get_collections().  Users may
        # replace it with relative Suite2p run paths before exporting.
        "batch_list": [],
        # The first discovered batch is selected by default.  A user-provided
        # relative Path is preserved by later get_collections() calls.
        "selected_batch": None,
    }
    datasets = {
        "mpci.times.npy": IBLDataset(
            ibl_object="mpci",
            attribute="times",
            extension="npy",
            namespace=None,
            export_fn=export_mpci_times,
            docstring=MPCI_TIMES_DOC,
        ),
        "mpci.badFrames.npy": IBLDataset(
            ibl_object="mpci",
            attribute="badFrames",
            extension="npy",
            namespace=None,
            export_fn=export_mpci_bad_frames,
            docstring=MPCI_BAD_FRAMES_DOC,
        ),
        "mpci.ROIActivityF.npy": IBLDataset(
            ibl_object="mpci",
            attribute="ROI_activity_F",
            extension="npy",
            namespace=None,
            export_fn=export_mpci_roi_activity_f,
            docstring=MPCI_ROI_ACTIVITY_F_DOC,
        ),
        "mpci.ROINeuropilActivityF.npy": IBLDataset(
            ibl_object="mpci",
            attribute="ROI_neuropil_activity_F",
            extension="npy",
            namespace=None,
            export_fn=export_mpci_roi_neuropil_activity_f,
            docstring=MPCI_ROI_NEUROPIL_ACTIVITY_F_DOC,
        ),
        "mpci.ROIActivityDeconvolved.npy": IBLDataset(
            ibl_object="mpci",
            attribute="ROI_activity_deconvolved",
            extension="npy",
            namespace=None,
            export_fn=export_mpci_roi_activity_deconvolved,
            docstring=MPCI_ROI_ACTIVITY_DECONVOLVED_DOC,
        ),
        "mpciROIs.mpciROITypes.npy": IBLDataset(
            ibl_object="mpciROIs",
            attribute="mpciROITypes",
            extension="npy",
            namespace=None,
            export_fn=export_mpci_roi_types,
            docstring=MPCI_ROI_TYPES_DOC,
        ),
        "mpciROIs.cellClassifier.npy": IBLDataset(
            ibl_object="mpciROIs",
            attribute="cellClassifier",
            extension="npy",
            namespace=None,
            export_fn=export_mpci_cell_classifier,
            docstring=MPCI_CELL_CLASSIFIER_DOC,
        ),
        "mpciROIs.masks.sparse_npz": IBLDataset(
            ibl_object="mpciROIs",
            attribute="masks",
            extension="sparse_npz",
            namespace=None,
            export_fn=export_mpci_roi_masks,
            docstring=MPCI_ROI_MASKS_DOC,
        ),
        "mpciROIs.stackPos.npy": IBLDataset(
            ibl_object="mpciROIs",
            attribute="stackPos",
            extension="npy",
            namespace=None,
            export_fn=export_mpci_stack_pos,
            docstring=MPCI_STACK_POS_DOC,
        ),
        "mpciMeanImage.images.npy": IBLDataset(
            ibl_object="mpciMeanImage",
            attribute="images",
            extension="npy",
            namespace=None,
            export_fn=export_mpci_mean_image,
            docstring=MPCI_MEAN_IMAGE_DOC,
        ),
    }

    def _handle_batching(self) -> bool:
        """Select a Suite2p run and store this session's frame range within it."""
        session_converter = self.session_converter
        current_session_key = (
            session_converter.modality.casefold(),
            session_converter.subject.casefold(),
            session_converter.date.replace("-", ""),
            session_converter.session_number,
        )

        if not self.batch_list:
            processed_subject_path = (
                session_converter.processed_data_root
                / session_converter.modality
                / session_converter.subject
            )
            if processed_subject_path.is_dir():
                for db_path in processed_subject_path.rglob("db.npy"):
                    run_path = db_path.parent
                    has_plane_folder = any(
                        path.is_dir()
                        and re.fullmatch(r"plane\d+", path.name, re.IGNORECASE)
                        for path in run_path.iterdir()
                    )
                    if not has_plane_folder:
                        continue

                    data_paths = np.load(db_path, allow_pickle=True).item()["data_path"]
                    if isinstance(data_paths, (str, Path)):
                        data_paths = [data_paths]

                    session_keys = []
                    for data_path in data_paths:
                        parts = [
                            part
                            for part in str(data_path).replace("\\", "/").split("/")
                            if part
                        ]
                        two_p_index = max(
                            i for i, part in enumerate(parts) if part.casefold() == "2p"
                        )
                        modality, subject, date, session_number = parts[
                            two_p_index : two_p_index + 4
                        ]
                        session_keys.append(
                            (
                                modality.casefold(),
                                subject.casefold(),
                                date.replace("-", ""),
                                int(session_number),
                            )
                        )

                    if session_keys.count(current_session_key) == 1:
                        self.batch_list.append(
                            run_path.relative_to(processed_subject_path)
                        )

        if self.selected_batch is None and self.batch_list:
            self.selected_batch = self.batch_list[0]
        if not self.batch_list:
            return False

        processed_subject_path = (
            session_converter.processed_data_root
            / session_converter.modality
            / session_converter.subject
        )
        run_path = processed_subject_path / self.selected_batch

        db_path = run_path / "db.npy"
        data_paths = np.load(db_path, allow_pickle=True).item()["data_path"]
        if isinstance(data_paths, (str, Path)):
            data_paths = [data_paths]

        session_keys = []
        raw_session_paths = []
        # Suite2p concatenates trace frames in db.npy data_path order, so the
        # same order defines the frame offsets for the selected session.
        for data_path in data_paths:
            parts = [part for part in str(data_path).replace("\\", "/").split("/") if part]
            two_p_index = max(i for i, part in enumerate(parts) if part.casefold() == "2p")
            modality, subject, date, session_number = parts[
                two_p_index : two_p_index + 4
            ]
            session_keys.append(
                (
                    modality.casefold(),
                    subject.casefold(),
                    date.replace("-", ""),
                    int(session_number),
                )
            )
            raw_session_paths.append(
                session_converter.raw_data_root
                / modality
                / subject
                / date.replace("-", "")
                / str(int(session_number))
            )
        if session_keys.count(current_session_key) != 1:
            raise ValueError(
                f"Current session {current_session_key} must occur exactly once in db data_path: "
                f"{session_keys}"
            )

        session_frame_counts = []
        # Count DI3 imaging frames for every raw session in the batch.  The
        # label interpretation mirrors _prepare_fov_timing().
        for raw_session_path in raw_session_paths:
            timing_path = raw_session_path / "timestamped_digital.csv"
            with timing_path.open(newline="") as stream:
                rows = list(csv.reader(stream))
            use_third_column = not any(
                len(row) > 1 and row[1].strip() == "DI3" for row in rows
            )
            frame_count = 0
            for row in rows:
                label = (
                    row[2].strip()
                    if use_third_column and len(row) > 2
                    else "" if use_third_column else row[1].strip()
                )
                if label == "DI3":
                    frame_count += 1
            session_frame_counts.append(frame_count)

        current_session_index = session_keys.index(current_session_key)
        self.run_path = run_path
        self.frame_start = sum(session_frame_counts[:current_session_index])
        self.frame_stop = (
            self.frame_start + session_frame_counts[current_session_index]
        )
        self.total_frame_count = sum(session_frame_counts)
        return True

    def _handle_planes(self) -> dict[int, dict]:
        """Return lightweight metadata for each FOV in the selected run."""
        planes = []
        for path in self.run_path.iterdir():
            match = re.fullmatch(r"plane(\d+)", path.name, re.IGNORECASE)
            if path.is_dir() and match:
                planes.append((int(match.group(1)), path))
        planes.sort()
        if not planes:
            raise FileNotFoundError(f"No planeX folders found in {self.run_path}")

        # FOVs are keyed by the Suite2p folder number (planeN -> FOV_NN), as in
        # IBL's mesoscope pipeline.  ops["iroi"] is not used because it repeats
        # across z-planes of the same ROI.  combined/stat["iplane"] uses the
        # same folder number.
        fovs = {}
        required_plane_files = ("F.npy", "Fneu.npy", "spks.npy", "stat.npy", "ops.npy")
        for plane_index, plane_path in planes:
            missing = [
                filename
                for filename in required_plane_files
                if not (plane_path / filename).is_file()
            ]
            if missing:
                raise FileNotFoundError(f"Missing files in {plane_path}: {missing}")
            ops = np.load(plane_path / "ops.npy", allow_pickle=True).item()
            fovs[plane_index] = {
                "plane_index": plane_index,
                "path": plane_path,
                "height": int(ops["Ly"]),
            }

        combined_path = self.run_path / "combined"
        if combined_path.is_dir():
            for filename in ("iscell.npy", "stat.npy"):
                if not (combined_path / filename).is_file():
                    raise FileNotFoundError(
                        f"Missing combined curation file: {combined_path / filename}"
                    )
            for fov in fovs.values():
                fov["iscell_path"] = combined_path / "iscell.npy"
                fov["stat_path"] = combined_path / "stat.npy"
        elif len(fovs) == 1:
            fov = next(iter(fovs.values()))
            iscell_path = fov["path"] / "iscell.npy"
            if not iscell_path.is_file():
                raise FileNotFoundError(f"Missing single-plane curation file: {iscell_path}")
            fov["iscell_path"] = iscell_path
            fov["stat_path"] = fov["path"] / "stat.npy"
        else:
            raise FileNotFoundError(
                "Multi-plane runs require combined/iscell.npy and combined/stat.npy"
            )

        return fovs

    def _prepare_fov_timing(self) -> int:
        """Store corrected mROI times and timing-derived bad frames."""
        fovs = self.fovs
        timing_path = self.session_converter.raw_path / "timestamped_digital.csv"
        with timing_path.open(newline="") as stream:
            rows = list(csv.reader(stream))

        use_third_column = not any(
            len(row) > 1 and row[1].strip() == "DI3" for row in rows
        )
        di3_times = []
        paired_di3_times = []
        none_times = []
        pending = None
        for row in rows:
            label = (
                row[2].strip()
                if use_third_column and len(row) > 2
                else "" if use_third_column else row[1].strip()
            )
            if label == "DI3":
                pending = float(row[0])
                di3_times.append(pending)
            elif label in {"", "None"} and pending is not None:
                paired_di3_times.append(pending)
                none_times.append(float(row[0]))
                pending = None

        di3_times = np.asarray(di3_times, dtype=np.float64)
        paired_di3_times = np.asarray(paired_di3_times, dtype=np.float64)
        cycle_widths = np.asarray(none_times, dtype=np.float64) - paired_di3_times
        timing_bad = np.zeros(di3_times.size, dtype=bool)

        # Preserve the common final-frame edge case where the last None edge is absent.
        if paired_di3_times.size == di3_times.size - 1:
            cycle_widths = np.append(cycle_widths, np.median(cycle_widths[-100:]))
            timing_bad[-1] = True

        # Line offsets accumulate in plane folder order, which is FOV key order.
        heights = [fovs[index]["height"] for index in sorted(fovs)]
        total_height = sum(heights)
        height_before = 0
        self.times_by_fov = {}
        self.timing_bad_frames_by_fov = {}
        for fov_index, height in zip(sorted(fovs), heights):
            corrected = di3_times + cycle_widths * height_before / total_height
            self.times_by_fov[fov_index] = (
                corrected - self.session_converter.session_time_zero
            )
            self.timing_bad_frames_by_fov[fov_index] = timing_bad.copy()
            height_before += height

        return di3_times.size

    def get_collections(self) -> list[Collection]:
        if not self._handle_batching():
            return []

        self.fovs = self._handle_planes()

        # Prepare FOV timestamps from only the requested raw session.
        # They correspond directly to the trace segment selected below.
        session_frame_count = self._prepare_fov_timing()
        expected_frame_count = self.frame_stop - self.frame_start
        if session_frame_count != expected_frame_count:
            raise ValueError(
                f"Current session timing contains {session_frame_count} DI3 frames, but "
                f"the selected batch count is {expected_frame_count}"
            )

        alf = Collection("alf")
        for fov_index in sorted(self.fovs):
            alf.collections.append(
                Collection(
                    "FOV",
                    index=fov_index,
                    datasets=[
                        self.datasets["mpci.times.npy"],
                        self.datasets["mpci.badFrames.npy"],
                        self.datasets["mpci.ROIActivityF.npy"],
                        self.datasets["mpci.ROINeuropilActivityF.npy"],
                        self.datasets["mpci.ROIActivityDeconvolved.npy"],
                        self.datasets["mpciROIs.mpciROITypes.npy"],
                        self.datasets["mpciROIs.cellClassifier.npy"],
                        self.datasets["mpciROIs.masks.sparse_npz"],
                        self.datasets["mpciROIs.stackPos.npy"],
                        self.datasets["mpciMeanImage.images.npy"],
                    ],
                )
            )
        return [alf]
