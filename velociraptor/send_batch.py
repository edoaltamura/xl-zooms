from tqdm import tqdm
import numpy as np
from h5py import File as h5file
import argparse
import os.path

parser = argparse.ArgumentParser()

parser.add_argument(
    '-d',
    '--directories',
    type=str,
    nargs='+',
    required=True
)

parser.add_argument(
    '-i',
    '--no-snapid',
    default=False,
    required=False,
    action='store_true'
)

args = parser.parse_args()

# Velociraptor invokes

modules = (
    "module purge\n"
    "module load cmake/3.18.1\n"
    "module load intel_comp/2020-update2\n"
    "module load intel_mpi/2020-update2\n"
    "module load ucx/1.8.1\n"
    "module load parmetis/4.0.3-64bit\n"
    "module load parallel_hdf5/1.10.6\n"
    "module load fftw/3.3.8cosma7\n"
    "module load gsl/2.5\n"
)

parameter_file = "../vrconfig_3dfofbound_subhalos_SO_hydro.cfg"
executable_path = "/cosma7/data/dp004/dc-alta2/xl-zooms/$switch_mode/VELOCIraptor-STF_hotgas_2020/stf"

epilog = (
    'echo "Job done, info follows."\n'
    'sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode\n'
)


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def make_sbatch_params(ntasks: int = 1, cpus_per_task: int = 28, run_name: str = 'VR-analysis'):
    return (
        "# !/bin/bash -l\n"
        f"# SBATCH --ntasks={ntasks}\n"
        f"# SBATCH --cpus-per-task={cpus_per_task}\n"
        "# SBATCH --exclusive\n"
        f"# SBATCH -J {run_name}\n"
        "# SBATCH -o ./out_files/%x.%J.vr.out\n"
        "# SBATCH -e ./out_files/%x.%J.vr.err\n"
        "# SBATCH -p cosma7\n"
        "# SBATCH -A dp004\n"
        "# SBATCH -t 72:00:00\n"
        f"export OMP_NUM_THREADS={ntasks}\n"
    )


def make_stf_invoke(input_file: str, output_file: str):
    return (
        f"{executable_path}"
        f" -I 2"
        f" -i {input_file.rstrip('.hdf5')}"
        f" -o {output_file.rstrip('.hdf5')}"
        f" -C {parameter_file}"
    )


for i, run_directory in enumerate(args.directories):
    snaps_path = os.path.join(run_directory, 'snapshots')
    catalogues_path = os.path.join(run_directory, 'stf')

    if not os.path.isdir(snaps_path):
        raise NotADirectoryError(f"Snapshot directory {snaps_path} not found.")

    snapshot_files = []
    snapshot_sizes = []
    stf_subdirs = []
    for file in tqdm(os.listdir(snaps_path), desc='Identify snapshots'):

        file_path = os.path.join(snaps_path, file)
        stf_subdir = os.path.join(catalogues_path, file.rstrip('.hdf5'))

        if os.path.isfile(file_path) and file.endswith('.hdf5'):

            if not args.no_snapid:
                with h5file(file_path, 'r') as f:
                    output_type = f['Header'].attrs['SelectOutput'].decode('ascii')
            else:
                output_type = 'Default'

            # Check if the file is snapshot (keep) or snipshot (skip)
            if output_type == 'Default':
                snapshot_files.append(file_path)
                snapshot_sizes.append(os.path.getsize(file_path))
                stf_subdirs.append(stf_subdir)

    number_snapshots = len(snapshot_files)
    snapshot_sizes = np.asarray(snapshot_sizes, dtype=np.int64)

    if number_snapshots == 0:
        raise FileNotFoundError(f"No snapshot file found in {snaps_path}")
    else:
        print((
            f"\nFound {number_snapshots:d} snapshots in directory "
            f"{snaps_path}.\n"
            f"Total file size = {sizeof_fmt(snapshot_sizes.sum())}\n"
            f"Average file size = {sizeof_fmt(snapshot_sizes.mean())}\n"
            f"Min/max file size = {sizeof_fmt(snapshot_sizes.min())}/{sizeof_fmt(snapshot_sizes.max())}\n"
        ))

    # Split tasks in jobs for 800 GB of input data each
    job_limit = 800 * 1024 * 1024 * 1024
    print(f"Input data limit: {sizeof_fmt(job_limit)} per batch.")

    number_splits = snapshot_sizes.sum() // job_limit + 1
    chunk_items = np.ones(number_splits + 1, dtype=np.int) * len(snapshot_sizes) // number_splits
    chunk_items[-1] = len(snapshot_sizes) % number_splits
    chunk_items = np.cumsum(chunk_items)
    print(chunk_items)
    split_indices = np.split(np.arange(number_snapshots), chunk_items)

    for i, split_batch in enumerate(split_indices):
        print((
            f"Batch {i + 1}/{len(split_indices)} | "
            f"Invoking VR on {len(split_batch)} snapshots. "
            f"Total batch size {sizeof_fmt(snapshot_sizes[split_batch].sum())}"
        ))

