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
    if not os.path.isfile(os.path.join(run_directory, "snap_redshifts.txt")):
        print(f"snap_redshifts.txt file not found. Nothing was compressed")
        return

    if not os.path.isdir(os.path.join(run_directory, "snapshots")):
        print(f"snapshots directory not found. Nothing was compressed")
        return

    if os.path.isfile(os.path.join(run_directory, "snapshots", "snipshots.zip")):
        print(f"snipshots.zip already exists. Nothing was compressed")
        return

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
    # Use len_dir_path to avoid including the internal directory structure inside zip archive
    archive_filename = os.path.join(run_directory, "snapshots", "snipshots.zip")
    len_dir_path = len(os.path.join(run_directory, "snapshots"))
    with zipfile.ZipFile(archive_filename, 'w', zipfile.ZIP_DEFLATED) as zip_handle:
        file_sizes = np.ones_like(snipshots_filenames, dtype=np.int64)
        for i in trange(len(snipshots_filenames), desc=f"Compression to snipshots.zip"):
            file = os.path.join(run_directory, 'snapshots', snipshots_filenames[i])
            size = os.path.getsize(file)
            file_sizes[i] = size
            zip_handle.write(file, file[len_dir_path:])
            os.remove(file)

    print(
        "Compression complete.\n"
        f"Minimum file size: {humanize_bytes(np.min(file_sizes))}\n"
        f"Maximum file size: {humanize_bytes(np.max(file_sizes))}\n"
        f"Total file size: {humanize_bytes(np.sum(file_sizes))}\n"
        f"Archive file size: {humanize_bytes(os.path.getsize(archive_filename))}"
    )


def remove_restart_files(run_directory: str):
    if os.path.isdir(os.path.join(run_directory, "restart")):
        shutil.rmtree(os.path.join(run_directory, "restart"))
    else:
        print("No restart directory found. Nothing was removed.")


def extract_snipshots(run_directory: str):
    archive_filename = os.path.join(run_directory, "snapshots", "snipshots.zip")
    dest_directory = os.path.join(run_directory, "snapshots", "snipshots")

    with zipfile.ZipFile(archive_filename, 'r') as zip_handle:
        contents = zip_handle.namelist()

        # Extract files in a loop - same implementation of self.extractall() in Lib/zipfile.
        for i in trange(len(contents), desc=f"Extracting from snipshots.zip"):
            zip_handle.extract(contents[i], dest_directory)


def remove_extracted_snipshots(run_directory: str):
    if os.path.isdir(os.path.join(run_directory, "snapshots", "snipshots")):
        shutil.rmtree(os.path.join(run_directory, "snapshots", "snipshots"))
    else:
        print("No extracted snipshots directory found. Nothing was removed.")


if __name__ == '__main__':

    runs = [
        "/cosma/home/dp004/dc-alta2/snap7/xl-zooms/hydro/L0300N0564_VR23_-8res_MinimumDistance_fixedAGNdT7.5_Nheat1_SNnobirth",
        "/cosma/home/dp004/dc-alta2/snap7/xl-zooms/hydro/L0300N0564_VR23_-8res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth",
        "/cosma/home/dp004/dc-alta2/snap7/xl-zooms/hydro/L0300N0564_VR23_-8res_MinimumDistance_fixedAGNdT8_Nheat1_SNnobirth",
        "/cosma/home/dp004/dc-alta2/snap7/xl-zooms/hydro/L0300N0564_VR23_-8res_MinimumDistance_fixedAGNdT9.5_Nheat1_SNnobirth",
        "/cosma/home/dp004/dc-alta2/snap7/xl-zooms/hydro/L0300N0564_VR23_-8res_MinimumDistance_fixedAGNdT9_Nheat1_SNnobirth",
    ]

    for working_run_directory in runs:
        print(working_run_directory)
        # remove_extracted_snipshots(working_run_directory)
        compress_snipshots(working_run_directory)
        remove_restart_files(working_run_directory)
        # extract_snipshots(working_run_directory)
