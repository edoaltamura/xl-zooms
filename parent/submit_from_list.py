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
        "-t param_files/default_sept.yml "
        "-l /cosma/home/dp004/dc-alta2/data7/xl-zooms/ics/particle_loads/masks_list.txt "
        "-s "
    ),
)

parser.add_argument(
    '-t',
    '--template',
    action='store',
    required=True,
    type=str
)

parser.add_argument(
    '-l',
    '--listfile',
    action='store',
    required=True,
    type=str
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
)
parser.add_argument(
    '-p',
    '--particle-load-library',
    action='store',
    default='.',
    required=False,
)

args = parser.parse_args()

try:
    from Generate_PL import ParticleLoad, comm_rank
except ImportError:
    try:
        if args.particle_load_library:
            sys.path.append(args.particle_load_library)
            from Generate_PL import ParticleLoad, comm_rank
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
            os.path.split(mask_filepath)[-1]
        )[0]
        try:
            detect_group_number = re.search('_VR(.+?)_', mask_name).group(1)
        except AttributeError:
            # Group Number not found in the original string
            detect_group_number = ''
        group_numbers.append(detect_group_number)
    print(group_numbers)
    return lines


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
    out_dir = get_output_directory()

    if not os.path.isfile(os.path.join(out_dir, os.path.basename(args.template))):
        copyfile(
            os.path.join(this_file_directory, args.template),
            os.path.join(out_dir, os.path.basename(args.template))
        )

    for mask_filepath in get_mask_paths_list():

        # Construct particle load parameter file name
        mask_name = os.path.splitext(
            os.path.split(mask_filepath)[-1]
        )[0]

        file_name = get_from_template('f_name').replace('FILENAME', str(mask_name))
        particle_load_paramfile = os.path.join(out_dir, f"{file_name}.yml")
        copyfile(os.path.join(out_dir, os.path.basename(args.template)), particle_load_paramfile)

        replace_pattern('PATH_TO_MASK', str(mask_filepath), particle_load_paramfile)
        replace_pattern('FILENAME', str(mask_name), particle_load_paramfile)

        if comm_rank == 0:
            print("ParticleLoad(particle_load_paramfile, only_calc_ntot=args.only_calc_ntot)")
        # ParticleLoad(particle_load_paramfile, only_calc_ntot=args.only_calc_ntot)

        if args.submit:
            old_cwd = os.getcwd()
            ic_submit_dir = os.path.join(
                get_from_template('ic_dir'),
                'ic_gen_submit_files',
                file_name
            )
            # os.chdir(ic_submit_dir)
            if comm_rank == 0:
                print(f"Submitting IC_Gen.x at {ic_submit_dir}")
            # subprocess.call(["sbatch", "submit.sh"])

            os.chdir(old_cwd)


if __name__ == '__main__':
    make_particle_load_from_list()
