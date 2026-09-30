"""Manual end-to-end export and ONE loading flow for NM023 2026-04-28."""

from alf_toolbox import OneSessionConverter, explain_dataset
from one.api import One, make_parquet_db



session = OneSessionConverter(
    "2P",
    "NM023",
    "20260428",
    1,
    raw_data_root="//10.233.25.135/FANCiNAS1/Data",
    processed_data_root="//10.233.25.135/FANCiNAS1/Processed",
    converter_root="D:/alftoolbox",
)

# List available exporters and their current settings.
print(session.exporters)

# Reassign the default candidate grids to show how settings are changed.
session.exporters.SparseNoiseExporter.sparse_noise_shapes = ((15, 15), (10, 30), (7, 27))

# Export the sparse-noise, mPCI, and wheel datasets available for this session.
session.export()

# Open the exported data with ONE.
make_parquet_db(session.one_root, hash_files=True)

# Regex mode can match unrevisioned local datasets when loading an object by
# its ALF parts; ONE's default wildcard mode currently cannot.
one = One(session.one_root, mode="local", wildcards=False)

eid = one.search(
    subject=session.subject,
    date_range=session.date,
    number=session.session_number,
)[0]

# Discover datasets and collections through ONE.
datasets = one.list_datasets(eid)
collections = one.list_collections(eid)
print(datasets)
print(collections)

# Show documentation for a dataset from each exporter.
explain_dataset("alf/FOV_00/mpci.times.npy")
explain_dataset("alf/_ibl_sparseNoise.times.npy")
explain_dataset("alf/_ibl_wheel.position.npy")

# Load each exporter's data through ONE's dataset and object interfaces.
mpci_times = one.load_dataset(eid, "alf/FOV_00/mpci.times.npy")
sparse_noise_times = one.load_dataset(eid, "alf/_ibl_sparseNoise.times.npy")
wheel_timestamps = one.load_dataset(eid, "alf/_ibl_wheel.timestamps.npy")
mpci = one.load_object(eid, "mpci", collection="alf/FOV_00")
sparse_noise = one.load_object(eid, "sparseNoise", collection="alf")
wheel = one.load_object(eid, "wheel", collection="alf")

# Both ALF collections can also be loaded as complete collections.
fov = one.load_collection(eid, "alf/FOV_00")
alf = one.load_collection(eid, "alf")

assert mpci_times is not None and "times" in mpci and fov
assert sparse_noise_times is not None and "times" in sparse_noise
assert wheel_timestamps is not None and "position" in wheel and "velocity" in wheel
assert alf is not None
