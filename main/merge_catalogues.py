import sys
import os.path
import pandas as pd

sys.path.append("..")

from register import (
    calibration_zooms,
    completed_runs,
    zooms_register,
    default_output_directory,
    DataframePickler,
)

mass_estimator = 'true'

catalogues_dir = os.path.join(default_output_directory, 'intermediate')

files = []

for f in os.listdir(catalogues_dir):
    if (
            f.endswith('.pkl') and
            mass_estimator in f and
            'register' not in f
    ):
        files.append(os.path.join(catalogues_dir, f))

catalogue = DataframePickler(files[0]).load_from_pickle()

for f in files[1:]:
    catalogue = pd.concat(
        [
            catalogue,
            DataframePickler(f).load_from_pickle()
        ],
        axis=1
    )

catalogue = catalogue.loc[:, ~catalogue.columns.duplicated()]
print(catalogue.head())
