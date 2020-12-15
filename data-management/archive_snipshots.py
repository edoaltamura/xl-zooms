import os
import zipfile
import shutil
import numpy as np


def human_readable_format(size, precision=2):
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB']
    suffix_index = 0
    while size > 1024:
        suffix_index += 1     # increment the index of the suffix
        size = size / 1024.0  # apply the division
    return "%.*f %d" % (precision, size, suffixes[suffix_index])


def zipdir(path: str, zip_handle: zipfile.ZipFile):
    # Get output type list
    output_list = np.genfromtxt(
        os.path.join(path, "snap_redshifts.txt"),
        delimiter=', ', dtype=str
    ).T[1]

    snipshot_index = np.where(output_list == 'Snipshot')[0]

    # Get filenames in output data directory
    output_data_directory = os.path.join(path, "snapshots")
    outfiles = [f for f in os.listdir(output_data_directory) if os.path.isfile(os.path.join(output_data_directory, f))]
    outfiles = [f for f in outfiles if f.endswith('.hdf5')]
    outfiles_index = [int(f[-9:-5]) for f in outfiles]

    # Sort filenames in data directory
    sort_key = np.argsort(outfiles_index)
    outfiles = np.asarray(outfiles)[sort_key]
    outfiles_index = np.asarray(outfiles_index)[sort_key]

    assert all(x < y for x, y in zip(outfiles_index[:-1], outfiles_index[1:])), (
        "Checking monotonicity of sorted indices for output files failed. "
        "The sequence of file indices is not strictly increasing. Check sorting algorithm."
    )

    snipshots_filenames = outfiles[snipshot_index]

    print(snipshots_filenames, snipshot_index)

    file_sizes = np.ones_like(snipshots_filenames, dtype=np.int64)
    for i, file in enumerate(snipshots_filenames):
        size = os.path.getsize(os.path.join(path, 'snapshots', file))
        print(human_readable_format(size))
        file_sizes[i] = size
        # zip_handle.write(file)
    print(np.sum(file_sizes))


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
