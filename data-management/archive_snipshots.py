import os
import zipfile
import shutil
import numpy as np


def zipdir(path: str, zip_handle: zipfile.ZipFile):
    # Get output type list
    output_list = np.genfromtxt(
        os.path.join(path, "snap_redshifts.txt"),
        delimiter=',', dtype=None
    ).T[1]

    print(output_list)

    # for file in output_list:
    #     zip_handle.write(file)


def archive_outputs(snap_directory: str):
    # Make sure to be in the correct working directory
    os.chdir(snap_directory)
    dir_basename = os.path.basename(os.path.normpath(os.getcwd()))
    assert 'snap' in dir_basename.lower(), "Check that what you are archiving is a snapshot directory."

    # Identify contents recursively and append to zip handle
    with zipfile.ZipFile(f'{dir_basename}.zip', 'w', zipfile.ZIP_DEFLATED) as zip_handle:
        zipdir(snap_directory, zip_handle)


def remove_restart_files(snap_directory: str):
    shutil.rmtree(os.path.join(snap_directory, "restart"))


if __name__ == '__main__':
    zipdir("/cosma/home/dp004/dc-alta2/snap7/xl-zooms/hydro/L0300N0564_VR3032_-8res_Isotropic", None)
