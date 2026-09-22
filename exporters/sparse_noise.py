from pathlib import Path
from ..core import *
import numpy as np

#Docstrings
DOCSTRING_passiveRFM_times_npy="Docstring text"



#Export implementation
def export_passiveRFM(source_paths,output_path):
    print("exporting",source_paths," to ",output_path)

    frame_times = np.loadtxt(source_paths["times_grid.csv"], delimiter=",", ndmin=1) #This is wrong because source_paths should be a list of Path objects
    np.save(output_path, np.ravel(frame_times).astype(np.float64, copy=False))
    return output_path


#Session aware organization
class SparseNoiseExporter:
    required_filenames=["SparseNoise_Log.bin", "times_grid.csv", "timestamped_digital.csv"]

    def get_collections(onesession):
        raw_path = onesession.raw_path
        assert all((raw_path / filename).is_file() for filename in SparseNoiseExporter.required_filenames)
        required_files = [raw_path / filename for filename in SparseNoiseExporter.required_filenames]


        alfcol=Collection('alf')
        datasets=[IBLDataset(
                    IBLobject="passiveRFM",
                    attribute="times",
                    extension="npy",
                    export_fn=export_passiveRFM,
                    docstring=DOCSTRING_passiveRFM_times_npy)]   

        alfcol.datasets.extend(datasets)

        return [alfcol]

    
