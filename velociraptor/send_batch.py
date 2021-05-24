from tqdm import tqdm
import numpy as np
from h5py import File as h5file
import argparse
import os.path
import subprocess

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
    '--check-snipshots',
    default=False,
    required=False,
    action='store_true'
)

parser.add_argument(
    '-s',
    '--submit',
    default=False,
    required=False,
    action='store_true'
)

parser.add_argument(
    '-l',
    '--limit-batch-gb',
    type=int,
    default=500,
    required=False,
)

parser.add_argument(
    '--with-gnu-parallel',
    default=False,
    required=False,
    action='store_true'
)

parser.add_argument(
    '--with-mpi',
    default=False,
    required=False,
    action='store_true'
)

parser.add_argument(
    '--stf-executable',
    type=str,
    default="/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/VELOCIraptor-STF_hotgas_2020/stf",
    required=False,
)

parser.add_argument(
    '-C',
    '--config',
    type=str,
    default="/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/VELOCIraptor-STF_hotgas_2020/stf",
    required=False,
)

parser.add_argument(
    '-p',
    '--slurm-queue',
    type=str,
    default="cosma7-prince",
    required=False,
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
    "module load gnu-parallel/20181122\n"
)

# parameter_file = "../vrconfig_3dfofbound_subhalos_SO_hydro.cfg"
parameter_file = "/cosma/home/dp004/dc-alta2/snap7/xl-zooms/hydro/adiabatic_vr_test/vrconfig_swift_.cfg"
# executable_path = "/cosma7/data/dp004/dc-alta2/xl-zooms/hydro/VELOCIraptor-STF_hotgas_2020/stf"
executable_path = "/cosma/home/dp004/dc-alta2/data7/xl-zooms/dmo/VELOCIraptor-STF/stf"
slurm_queue = "cosma7-prince"

epilog = (
    '\necho "Job done, info follows."\n'
    'sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode\n'
)


def sizeof_fmt(num, suffix='B') -> str:
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def make_sbatch_params(ntasks: int = 28, cpus_per_task: int = 1, run_name: str = 'VR-analysis') -> str:
    sbatch_params = (
        f"#!/bin/bash -l\n"
        f"#SBATCH --ntasks={ntasks}\n"
        f"#SBATCH --cpus-per-task={cpus_per_task}\n"
        f"#SBATCH --exclusive\n"
        f"#SBATCH -J {run_name}\n"
        f"#SBATCH -o ./out_files/%x.%J.vr.out\n"
        f"#SBATCH -e ./out_files/%x.%J.vr.err\n"
        f"#SBATCH -p {slurm_queue}\n"
        f"#SBATCH -A dp004\n"
        f"#SBATCH -t 72:00:00\n"
    )

    if not args.with_gnu_parallel and not args.with_mpi:
        sbatch_params += f"export OMP_NUM_THREADS={cpus_per_task}\n"

    return sbatch_params


def make_stf_invoke(input_file: str, output_file: str) -> str:
    _input_file = input_file
    if input_file.endswith('.hdf5'):
        _input_file = input_file[:-5]

    _output_file = output_file
    if input_file.endswith('.hdf5'):
        _output_file = output_file[:-5]

    stf_invoke = (
        f"{executable_path}"
        f" -I 2"
        f" -i {_input_file}"
        f" -o {_output_file}"
        f" -C {parameter_file}"
    )

    if args.with_mpi:
        stf_invoke = "mpirun -np $SLURM_NTASKS " + stf_invoke

    return stf_invoke


def gnu_parallel_wrap(input_commands: list, options: str = ''):
    command_array = "commands=(\n"
    for cmd in input_commands:
        command_array += f"\"{cmd}\" \n"
    command_array += ")\n\n"

    gnu_invoke = "printf \'%s\\n\' \"${commands[@]}\" | parallel --jobs $SLURM_NTASKS "
    gnu_invoke += f"{options:s}\n\n"

    return command_array + gnu_invoke


for i, run_directory in enumerate(args.directories):
    snaps_path = os.path.join(run_directory, 'snapshots')
    catalogues_path = os.path.join(run_directory, 'stf')

    if not os.path.isdir(catalogues_path):
        os.mkdir(catalogues_path)

    if not os.path.isdir(snaps_path):
        raise NotADirectoryError(f"Snapshot directory {snaps_path} not found.")

    snapshot_files = []
    snapshot_sizes = []
    stf_subdirs = []
    snapshot_numbers_sort = []
    for file in tqdm(os.listdir(snaps_path), desc='Identify snapshots'):

        file_path = os.path.join(snaps_path, file)
        stf_subdir = os.path.join(catalogues_path, file[:-5])

        # Check if stf subdirectory has properties file
        has_properties = False
        if os.path.isdir(stf_subdir) and len(os.listdir(stf_subdir)) > 0:
            for filename in os.listdir(stf_subdir):
                if filename.endswith('.properties'):
                    has_properties = True

        if os.path.isfile(file_path) and file.endswith('.hdf5') and not has_properties:

            if args.check_snipshots:
                with h5file(file_path, 'r') as f:
                    output_type = f['Header'].attrs['SelectOutput'].decode('ascii')
            else:
                output_type = 'Default'

            # Check if the file is snapshot (keep) or snipshot (skip)
            if output_type == 'Default':
                snapshot_files.append(file_path)
                snapshot_sizes.append(os.path.getsize(file_path))
                stf_subdirs.append(stf_subdir)
                snapshot_numbers_sort.append(int(file_path[-9:-5]))

    number_snapshots = len(snapshot_files)

    # Sort snapshots by their output number
    assert len(snapshot_numbers_sort) == len(set(snapshot_numbers_sort)), (
        "Snapshot numbers have duplicates!"
    )

    snapshot_numbers_sort = np.asarray(snapshot_numbers_sort, dtype=np.int32)
    sort_key = np.argsort(snapshot_numbers_sort)
    snapshot_sizes = np.asarray(snapshot_sizes, dtype=np.int64)[sort_key]
    snapshot_files = np.asarray(snapshot_files, dtype=np.str)[sort_key]
    stf_subdirs = np.asarray(stf_subdirs, dtype=np.str)[sort_key]

    if number_snapshots == 0:
        if len(os.listdir(snaps_path)) == 0:
            raise FileNotFoundError(f"No snapshots file found in {snaps_path}")
        else:
            raise RuntimeError((
                f"Snapshot directory {snaps_path} contains "
                f"{len(os.listdir(snaps_path)):d} elements, "
                f"but none matched the search criteria."
            ))
    else:
        print((
            f"\nFound {number_snapshots:d} snapshots in directory "
            f"{snaps_path}.\n"
            f"Total file size = {sizeof_fmt(snapshot_sizes.sum())}\n"
            f"Average file size = {sizeof_fmt(snapshot_sizes.mean())}\n"
            f"Min/max file size = {sizeof_fmt(snapshot_sizes.min())}/{sizeof_fmt(snapshot_sizes.max())}\n"
        ))

    # Split tasks in jobs for 500 GB of input data each
    job_limit = args.limit_batch_gb * 1024 * 1024 * 1024
    print(f"Input data limit: {sizeof_fmt(job_limit)} per batch.")

    number_splits = snapshot_sizes.sum() // job_limit + 1
    chunk_items = np.ones(number_splits + 1, dtype=np.int) * len(snapshot_sizes) // number_splits
    chunk_items[-1] = len(snapshot_sizes) % number_splits
    chunk_items = np.cumsum(chunk_items)

    split_indices = np.split(np.arange(number_snapshots), chunk_items)
    split_indices = [split_batch for split_batch in split_indices if len(split_batch) > 0]
    number_batches = len(split_indices)

    for i, split_batch in enumerate(split_indices):

        print((
            f"Batch {i + 1:02d}/{number_batches:02d} | "
            f"Invoking VR on {len(split_batch)} snapshots. "
            f"Total batch size {sizeof_fmt(snapshot_sizes[split_batch].sum())}\n"
            "Snapshot numbers in this batch:"
        ))

        slurm_file = os.path.join(
            run_directory,
            f"vr_batch_{i:02d}.slurm"
        )
        with open(slurm_file, "w") as submit_file:

            print(
                make_sbatch_params(
                    run_name=f"VR_batch_{i:02d}_{os.path.basename(run_directory)}"
                ),
                file=submit_file
            )
            print(modules, file=submit_file)

            # If using GNU parallel, wrap commands into a bash array first
            if args.with_gnu_parallel:
                commands_to_gnu_parallel = list()

            for split_batch_item in split_batch:

                if not os.path.isdir(stf_subdirs[split_batch_item]):
                    os.mkdir(stf_subdirs[split_batch_item])

                snap_number = snapshot_files[split_batch_item][-9:-5]
                print(snap_number, end=' ')

                stf_invoke = make_stf_invoke(
                    input_file=snapshot_files[split_batch_item],
                    output_file=os.path.join(
                        stf_subdirs[split_batch_item],
                        os.path.basename(snapshot_files[split_batch_item])
                    )
                )

                if args.with_gnu_parallel:
                    commands_to_gnu_parallel.append(stf_invoke)
                else:
                    print(stf_invoke, file=submit_file)

            if args.with_gnu_parallel:
                print(
                    gnu_parallel_wrap(commands_to_gnu_parallel),
                    file=submit_file
                )

            print(epilog, file=submit_file)
            print(end='\n\n')

        if args.submit:
            print((
                f"Submitting {os.path.basename(slurm_file)} to the queue...\n"
                f"cwd >> {os.path.dirname(slurm_file)}\n"
                f"cmd >> {' '.join(['sbatch', os.path.basename(slurm_file)])}\n\n"
            ))
            p = subprocess.Popen(
                ['sbatch', os.path.basename(slurm_file)],
                cwd=os.path.dirname(slurm_file)
            )
            p.wait()
