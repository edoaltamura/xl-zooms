import sys
import os.path
import pandas as pd

sys.path.append("..")

from register import (
    default_output_directory,
    zooms_register,
    args,
    DataframePickler,
name_list
)

catalogues_dir = os.path.join(default_output_directory, 'intermediate')

files = []

for f in os.listdir(catalogues_dir):
    if (
            f.endswith('.pkl') and
            f'{args.redshift_index:04d}' in f and
            'register' not in f
    ):
        files.append(os.path.join(catalogues_dir, f))

    # elif 'vrproperties' in f or 'spherical_overdensities' in f:
    #     files.append(os.path.join(catalogues_dir, f))

catalogue = DataframePickler(files[0]).load_from_pickle()

for f in files[1:]:
    catalogue = pd.concat(
        [
            catalogue,
            DataframePickler(f).load_from_pickle()
        ],
        axis=1
    )

# Remove duplicate columns
catalogue = catalogue.loc[:, ~catalogue.columns.duplicated()]
print(catalogue.columns)
print(catalogue.head())


def select_runs():
    _zooms_register = []
    for keyword in args.keywords:
        for zoom in zooms_register:
            if keyword in zoom.run_name and zoom not in _zooms_register:
                _zooms_register.append(zoom)

    _name_list = [zoom.run_name for zoom in _zooms_register]

    return _name_list

models = []
for n in name_list:
    n = ''.join(n.split('_')[2:])
    models.append(n)
models = set(models)

print(models)