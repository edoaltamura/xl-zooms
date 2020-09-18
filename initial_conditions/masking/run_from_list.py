import argparse
import os
import re
from shutil import copyfile
from numpy import loadtxt, ndarray
from yaml import load

from make_mask import MakeMask

this_file_directory = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--template', action='store', type=str)
parser.add_argument('--listfile', action='store', type=str)
args = parser.parse_args()


def get_output_dir_from_template() -> str:
    params = load(open(args.template))
    output_dir = params['output_dir']
    if not os.path.isdir(output_dir):
        raise OSError(f"The specified output directory does not exist. Trying to save to {output_dir}")
    return output_dir


def get_run_name_from_template() -> str:
    params = load(open(args.template))
    return params['fname']


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


def make_masks_from_list() -> None:
    for group_number in get_group_numbers_list():

        mask_name = get_run_name_from_template().replace('GROUPNUMBER', str(group_number))
        mask_file = os.path.join(get_output_dir_from_template(), f"{mask_name}.hdf5")
        copyfile(os.path.join(this_file_directory, args.template), mask_file)
        replace_pattern('GROUPNUMBER', str(group_number), mask_file)
        sort_key = get_mass_sort_key()
        if '500' in sort_key:
            replace_pattern('SORTM200', '0', mask_file)
            replace_pattern('SORTM500', '1', mask_file)
        elif '200' in sort_key:
            replace_pattern('SORTM200', '1', mask_file)
            replace_pattern('SORTM500', '0', mask_file)
        mask = MakeMask(mask_file)


if __name__ == '__main__':
    make_masks_from_list()
