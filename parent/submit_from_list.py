import argparse
import os
import sys
import re
from shutil import copyfile
from numpy import ndarray
from yaml import load
from typing import List
import subprocess

this_file_directory = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    description=(
        "Generates template and submission files for running depositing the particle load."
    ),
    epilog=(
        "Example usage: "
        "mpirun -np 28 python3 submit_from_list.py "
        "-t /cosma/home/dp004/dc-alta2/data7/xl-zooms/ics/particle_loads/template_-8res.yml"
        "-l /cosma/home/dp004/dc-alta2/data7/xl-zooms/ics/particle_loads/masks_list.txt "
        "-p /cosma/home/dp004/dc-alta2/make_particle_load/particle_load/ "
        "-s "
    ),
)

parser.add_argument(
    '-t',
    '--template',
    action='store',
    required=True,
    type=str,
    help=(
        "The master parameter file to use as a template to generate mask-specific ones. "
        "It usually contains the hot keywords PATH_TO_MASK and FILENAME, which can be replaced "
        "with mask-dependent values."
    )
)

parser.add_argument(
    '-l',
    '--listfile',
    action='store',
    required=True,
    type=str,
    help=(
        "The file with the list of full paths to the mask files that are to be handled. "
        "The file paths are required to end with the file name with the correct `.hdf5` extension. "
        "The base-name of the masks files is used to replace the hot keywords PATH_TO_MASK and "
        "FILENAME in the template."
    )
)

parser.add_argument(
    '-o',
    '--only-calc-ntot',
    action='store_true',
    default=False,
    required=False,
)

parser.add_argument(
    '-s',
    '--submit',
    action='store_true',
    default=False,
    required=False,
    help=(
        "If activated, the program automatically executes the command `sbatch submit.sh` and launches the "
        "`IC_Gen.x` code for generating initial conditions. NOTE: all particle load in the list will be submitted "
        "to the SLURM batch system as individual jobs."
    )
)

parser.add_argument(
    '-p',
    '--particle-load-library',
    action='store',
    default='.',
    required=False,
    help=(
        "If this script is not located in the same directory as the `Generate_PL.py` code, you can import "
        "the code as an external library by specifying the full path to the `Generate_PL.py` file."
    )
)

parser.add_argument(
    '-d',
    '--dry',
    action='store_true',
    default=False,
    required=False,
    help=(
        "Use this option to produce dry runs, where the `ParticleLoad` class is deactivated, as well as the "
        "functionality for submitting jobs to the queue automatically, i.e. overrides --submit. Use this for "
        "testing purposes."
    )
)

args = parser.parse_args()

try:
    from Generate_PL import ParticleLoad, comm, comm_rank
except ImportError:
    try:
        if args.particle_load_library:
            sys.path.append(
                os.path.split(args.particle_load_library)[0]
            )
            from Generate_PL import ParticleLoad, comm, comm_rank
        else:
            raise Exception("The --particle-load-library argument is needed to import Generate_PL.py.")
    except ImportError:
        raise Exception("Make sure you have added the `Generate_PL.py` module directory to your $PYTHONPATH.")


def get_mask_paths_list() -> List[str]:
    with open(args.listfile) as f:
        lines = f.read().splitlines()

    group_numbers = []

    for line in lines:
        assert line.endswith('.hdf5'), f"Extension of the mask file {line} is ambiguous. File path must end in `.hdf5`."

        mask_name = os.path.splitext(
            os.path.split(line)[-1]
        )[0]
        group_numbers.append(int(mask_name.split('_VR')[-1]))

    lines_sorted = [x for _, x in sorted(zip(group_numbers, lines))]
    return lines_sorted


def get_output_directory() -> str:
    return os.path.split(args.listfile)[0]


def replace_pattern(pattern: str, replacement: str, filepath: str):
    with open(filepath, "r") as sources:
        lines = sources.readlines()
    with open(filepath, "w") as sources:
        for line in lines:
            sources.write(re.sub(pattern, replacement, line))


def get_from_template(parameter: str) -> str:
    params = load(open(args.template))
    return params[parameter]


def make_particle_load_from_list() -> None:

    if comm_rank == 0:
        out_dir = get_output_directory()
        if not os.path.isfile(os.path.join(out_dir, os.path.basename(args.template))):
            copyfile(
                os.path.join(this_file_directory, args.template),
                os.path.join(out_dir, os.path.basename(args.template))
            )
        mask_paths_list = get_mask_paths_list()
    else:
        out_dir = None
        mask_paths_list = None

    out_dir = comm.bcast(out_dir, root=0)
    mask_paths_list = comm.bcast(mask_paths_list, root=0)

    for mask_filepath in mask_paths_list:

        if comm_rank == 0:

            # Construct particle load parameter file name
            mask_name = os.path.splitext(
                os.path.split(mask_filepath)[-1]
            )[0]

            file_name = get_from_template('f_name').replace('FILENAME', str(mask_name))
            particle_load_paramfile = os.path.join(out_dir, f"{file_name}.yml")
            copyfile(os.path.join(out_dir, os.path.basename(args.template)), particle_load_paramfile)

            replace_pattern('PATH_TO_MASK', str(mask_filepath), particle_load_paramfile)
            replace_pattern('FILENAME', str(mask_name), particle_load_paramfile)

            print(f"ParticleLoad({particle_load_paramfile}, only_calc_ntot={args.only_calc_ntot})")

        else:
            particle_load_paramfile = None
            file_name = None

        particle_load_paramfile = comm.bcast(particle_load_paramfile, root=0)
        file_name = comm.bcast(file_name, root=0)

        if not args.dry:
            ParticleLoad(particle_load_paramfile, only_calc_ntot=args.only_calc_ntot)

        if args.submit and comm_rank == 0:
            old_cwd = os.getcwd()
            ic_submit_dir = os.path.join(
                get_from_template('ic_dir'),
                'ic_gen_submit_files',
                file_name
            )
            print(f"Submitting IC_Gen.x at {ic_submit_dir}")
            if not args.dry:
                os.chdir(ic_submit_dir)
                subprocess.call(["sbatch", "submit.sh"])

            os.chdir(old_cwd)


if __name__ == '__main__':
    make_particle_load_from_list()
