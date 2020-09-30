import argparse
import os
import re
import sys
from shutil import copyfile
from numpy import loadtxt, ndarray
import yaml
import random
from warnings import warn

this_file_directory = os.path.dirname(__file__)

parser = argparse.ArgumentParser(
    description=(
        "Generates masks for setting up initial conditions and allows object reselection."
    ),
    epilog=(
        "Example usage: python3 reselect_topology.py \ \n"
        "-t ~/xl-zooms/initial-conditions/make_mask/param_files/sample_list.yml \ \n"
        "-l ~/data7/xl-zooms/ics/masks/groupnumbers_defaultSept.txt \ \n"
        "-r ~/data7/xl-zooms/ics/masks/mass_bins_repository.yml"
    ),
)
parser.add_argument('-t', '--template', action='store', type=str, required=True)
parser.add_argument('-l', '--listfile', action='store', type=str, required=True)
parser.add_argument('-r', '--repository', action='store', type=str, required=True)
parser.add_argument(
    '-m',
    '--make-mask',
    action='store',
    type=str,
    required=False,
    default="/cosma/home/dp004/dc-alta2/make_particle_load/make_mask",
    help="Path to the `make_mask.py` module, used in the initial-conditions pipeline.",
)
args = parser.parse_args()

sys.path.append(args.make_mask)

try:
    from make_mask import MakeMask
except ImportError:
    raise Exception("Make sure you have added the `make_mask.py` module directory to your $PYTHONPATH.")
except:
    raise Exception(
        "Something else has gone wrong with importing the MakeMask class. "
        "Check that `make_mask.py` is set-up correctly and all its dependencies are in correct working order."
    )


def get_output_dir_from_template() -> str:
    params = yaml.load(open(args.template))
    output_dir = params['output_dir']
    if not os.path.isdir(output_dir):
        raise OSError(f"The specified output directory does not exist. Trying to save to {output_dir}")
    return output_dir


def get_run_name_from_template() -> str:
    with open(args.template, "r") as handle:
        params = yaml.load(handle, Loader=yaml.Loader)
        return params['fname']


def get_selection_repository() -> dict:
    with open(args.repository, "r") as handle:
        return yaml.load(handle, Loader=yaml.Loader)


def get_group_numbers_list() -> ndarray:
    return loadtxt(args.listfile).astype(int)


def get_mass_sort_key() -> str:
    with open(args.listfile, "r") as selection:
        lines = selection.readlines()
        for line in lines:
            if 'mass_sort' in line:
                sort_key = line.split()[-1]
                break
    assert '500' in sort_key or '200' in sort_key, ("Mass sort key returned unexpected value.",
                                                    f"Expected `M_200crit` or `M_500crit`, got {sort_key}")
    return sort_key


def replace_pattern(pattern: str, replacement: str, filepath: str):
    with open(filepath, "r") as sources:
        lines = sources.readlines()
    with open(filepath, "w") as sources:
        for line in lines:
            sources.write(re.sub(pattern, replacement, line))

def launch_mask(group_number: int, out_dir: str) -> None:

    if not os.path.isfile(os.path.join(out_dir, os.path.basename(args.template))):
        copyfile(
            os.path.join(this_file_directory, args.template),
            os.path.join(out_dir, os.path.basename(args.template))
        )
    mask_name = get_run_name_from_template().replace('GROUPNUMBER', str(group_number))
    mask_paramfile = os.path.join(out_dir, f"{mask_name}.yml")
    copyfile(os.path.join(out_dir, os.path.basename(args.template)), mask_paramfile)
    replace_pattern('GROUPNUMBER', str(group_number), mask_paramfile)
    sort_key = get_mass_sort_key()
    if '500' in sort_key:
        replace_pattern('SORTM200', '0', mask_paramfile)
        replace_pattern('SORTM500', '1', mask_paramfile)
    elif '200' in sort_key:
        replace_pattern('SORTM200', '1', mask_paramfile)
        replace_pattern('SORTM500', '0', mask_paramfile)
    mask = MakeMask(mask_paramfile)


if __name__ == '__main__':

    out_dir = get_output_dir_from_template()
    initial_selection_list = get_group_numbers_list()
    repository = get_selection_repository()
    final_selection_list = []

    with open(f"{out_dir}/groupnumbers_resampled.log", "w") as log:
        print("This file contains a log of the object indices with rejected and accepted masks.", file=log)
        print("-------------------------------------------------------------------------------.", file=log)

        for idx_bin, mass_bin in enumerate(repository):

            print(f"Examining selection repository: {mass_bin}")
            print(f"Examining selection repository: {mass_bin}", file=log)
            repository_bin_list = repository[mass_bin]['index_list']
            initial_bin_list = []

            for group_number in initial_selection_list:
                if group_number in repository_bin_list:
                    initial_bin_list.append(group_number)

            # Take away the initial random selection from the repository
            # If resampling needed, this won't repeat the same objects
            repository_bin_list = [x for x in repository_bin_list if x not in initial_bin_list]
            good_index_list = []
            bad_index_list = []

            for group_number in initial_bin_list:

                print(f"\tMasking object index {group_number} [initial selection]")
                print(f"\tMasking object index {group_number} [initial selection]", file=log)
                launch_mask(group_number, out_dir)
                is_ok_query = bool(input("\tEnter 1 to accept, 0 to sample a different object: "))
                print(f"\tEnter 1 to accept, 0 to sample a different object: {is_ok_query}", file=log)

                if is_ok_query:
                    good_index_list.append(group_number)
                else:
                    bad_index_list.append(group_number)

                    while True:

                        # If no more group numbers are left in the list to sample from, warn and do this manually
                        if len(repository_bin_list) == 0:
                            list_is_empty = (
                                "There are no group numbers to randomly sample from in this mass bin. "
                                "Presumably, some masks have been produced of objects in this mass bin - "
                                "check these masks in the group number dump-file and perform a super-extrusion "
                                "on the mask you wish to generate initial conditions from."
                            )
                            warn(list_is_empty)
                            print(list_is_empty, file=log)
                            break

                        new_group_number = random.sample(repository_bin_list, 1)
                        print(f"\tMasking object index {new_group_number} [resampled]")
                        print(f"\tMasking object index {new_group_number} [resampled]", file=log)
                        launch_mask(new_group_number, out_dir)
                        is_ok_query = bool(input("\tEnter 1 to accept, 0 to sample a different object: "))
                        print(f"\tEnter 1 to accept, 0 to sample a different object: {is_ok_query}", file=log)

                        if is_ok_query:
                            good_index_list.append(new_group_number)
                            break
                        else:
                            bad_index_list.append(new_group_number)
                            repository_bin_list = [x for x in repository_bin_list if x != new_group_number]
                            continue

            print("\tReport from this bin:")
            print("\t- group numbers from rejected masks: ", bad_index_list)
            print("\t- group numbers from accepted masks: ", good_index_list)
            print("\tReport from this bin:", file=log)
            print("\t- group numbers from rejected masks: ", bad_index_list, file=log)
            print("\t- group numbers from accepted masks: ", good_index_list, file=log)

            final_selection_list += good_index_list

        final_selection_list.sort()

        # Print to txt file
        with open(f"{args.listfile.replace('.txt', '_resampled.txt')}", "w") as text_file:
            print("# Halo index:", file=text_file)
            for i in final_selection_list:
                print(f"{i:d}", file=text_file)






