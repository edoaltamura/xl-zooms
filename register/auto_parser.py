# -*- coding: utf-8 -*-
"""Argument parser and CLI

This file defines the `ArgumentParser` instance for the command-line interface
of the analysis pipeline.
"""

import argparse
import os.path

from .revision import get_git_full
from .plotstyle import set_mpl_backend

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument(
    '-k',
    '--keywords',
    type=str,
    default=['L'],
    nargs='+',
    required=False
)

parser.add_argument(
    '-e',
    '--observ-errorbars',
    default=False,
    required=False,
    action='store_true'
)

parser.add_argument(
    '-z',
    '--redshift-index',
    type=int,
    default=36,
    required=False,
    choices=list(range(37))
)

parser.add_argument(
    '-m',
    '--mass-estimator',
    type=str.lower,
    default='true',
    required=False,
    choices=['true', 'hse', 'spec']
)

parser.add_argument(
    '-q',
    '--quiet',
    default=False,
    required=False,
    action='store_true'
)

parser.add_argument(
    '-d',
    '--debug',
    default=False,
    required=False,
    action='store_true'
)

parser.add_argument(
    '-r',
    '--refresh',
    default=False,
    required=False,
    action='store_true'
)

parser.add_argument(
    '-a',
    '--aperture-percent',
    type=int,
    default=100,
    required=False,
    choices=list(range(1, 400))
)

parser.add_argument(
    '-s',
    '--snapshot-number',
    type=int,
    required=True,
    default=36
)

parser.add_argument(
    '-b',
    '--run-directory',
    type=str,
    required=True,
    default=(
        '/cosma/home/dp004/dc-alta2/data6/xl-zooms/hydro/'
        'L0300N0564_VR18_+1res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth'
    )
)

# Note: you can still add routine-specific arguments after calling this
# The final version of the `args` objects will be created when calling
# `args = parser.parse_args()`
args = parser.parse_known_args()[0]


def find_files():
    s = ''
    for file in os.listdir(os.path.join(args.run_directory, 'snapshots')):
        if file.endswith(f"_{args.snapshot_number:04d}.hdf5"):
            s = os.path.join(args.run_directory, 'snapshots', file)
            break

    c = ''
    for subdir in os.listdir(os.path.join(args.run_directory, 'stf')):
        if subdir.endswith(f"_{args.snapshot_number:04d}"):
            for file in os.listdir(os.path.join(args.run_directory, 'stf', subdir)):
                if file.endswith(f"_{args.snapshot_number:04d}.properties"):
                    c = os.path.join(args.run_directory, 'stf', subdir, file)
                    break

    assert s and c

    return s, c

if not args.quiet:

    print("eagle-xl project ~ zoom-assisted calibration program".upper(), end='\n\n')

    print('Git revision:', *get_git_full(), sep='\n', end='\n\n')

    for parsed_argument in vars(args):
        print(f"[parser] {parsed_argument} = {getattr(args, parsed_argument)}")

# Set matplotlib backend depending on use
set_mpl_backend(args.quiet)
