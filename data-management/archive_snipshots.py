import os
import zipfile
import shutil
import numpy as np
from tqdm import trange


def humanize_bytes(num, suffix='B'):
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Y', suffix)


def compress_snipshots(run_directory: str):
    # Get output type list
    output_list = np.genfromtxt(
        os.path.join(run_directory, "snap_redshifts.txt"),
        delimiter=', ', dtype=str
    ).T[1]

    snipshot_index = np.where(output_list == 'Snipshot')[0]

    # Get filenames in output data directory
    output_data_directory = os.path.join(run_directory, "snapshots")
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

    # Filter snipshot filenames
    snipshots_filenames = outfiles[snipshot_index]
    print(
        f"Found {len(snipshots_filenames):d} snipshot files.\n"
        f"Found {(len(outfiles) - len(snipshots_filenames)):d} snapshot files."
    )

    # Compress snipshots and gather file sizes for report
    archive_filename = os.path.join(run_directory, "snapshots", "snipshots.zip")
    with zipfile.ZipFile(archive_filename, 'w', zipfile.ZIP_DEFLATED) as zip_handle:
        file_sizes = np.ones_like(snipshots_filenames, dtype=np.int64)
        for i in trange(len(snipshots_filenames), desc=f"Compression to snipshots.zip"):
            file = os.path.join(run_directory, 'snapshots', snipshots_filenames[i])
            size = os.path.getsize(file)
            file_sizes[i] = size
            zip_handle.write(file)
            os.remove(file)

    print(
        "Compression complete.\n"
        f"Minimum file size: {humanize_bytes(np.min(file_sizes))}\n"
        f"Maximum file size: {humanize_bytes(np.max(file_sizes))}\n"
        f"Total file size: {humanize_bytes(np.sum(file_sizes))}\n"
        f"Archive file size: {humanize_bytes(os.path.getsize(archive_filename))}\n"
    )


def remove_restart_files(run_directory: str):
    if os.path.isdir(os.path.join(run_directory, "restart")):
        shutil.rmtree(os.path.join(run_directory, "restart"))
    else:
        print("No restart directory found. Nothing was removed.")


def extract_snipshots(run_directory: str):
    archive_filename = os.path.join(run_directory, "snapshots", "snipshots.zip")
    with zipfile.ZipFile(archive_filename, 'r') as zip_handle:
        zip_handle.extractall(os.path.join(run_directory, "snapshots", "snipshots"))

def remove_extracted_snipshots(run_directory: str):
    if os.path.isdir(os.path.join(run_directory, "snapshots", "snipshots")):
        shutil.rmtree(os.path.join(run_directory, "snapshots", "snipshots"))
    else:
        print("No restart directory found. Nothing was removed.")


if __name__ == '__main__':
    remove_extracted_snipshots("/cosma/home/dp004/dc-alta2/snap7/xl-zooms/hydro/L0300N0564_VR3032_-8res_Isotropic")
    compress_snipshots("/cosma/home/dp004/dc-alta2/snap7/xl-zooms/hydro/L0300N0564_VR3032_-8res_Isotropic")
    remove_restart_files("/cosma/home/dp004/dc-alta2/snap7/xl-zooms/hydro/L0300N0564_VR3032_-8res_Isotropic")
    extract_snipshots("/cosma/home/dp004/dc-alta2/snap7/xl-zooms/hydro/L0300N0564_VR3032_-8res_Isotropic")
