# -*- coding: utf-8 -*-
"""Argument parser and CLI

This file defines the `ArgumentParser` instance for the command-line interface
of the analysis pipeline.
"""

import argparse
import os.path

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
    '-r',
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
    '-refresh',
    '--refresh-catalogue',
    default=False,
    required=False,
    action='store_true'
)

# Note: you can still add routine-specific arguments after calling this
# The final version of the `args` objects will be created when calling
# `args = parser.parse_args()`
args = parser.parse_known_args()[0]

# Set matplotlib backend depending on use
import matplotlib

mpl_backend = 'Agg' if args.quiet else 'TkAgg'
matplotlib.use(mpl_backend)

# Apply the matplotlib stylesheet
import matplotlib.pyplot as plt
matplotlib_stylesheet = os.path.join(
    os.path.dirname(__file__),
    'mnras.mplstyle'
)
try:
    plt.style.use(matplotlib_stylesheet)
except (FileNotFoundError, OSError):
    print('Could not find the mnras.mplstyle style-sheet.')

if not args.quiet:
    for parsed_argument in vars(args):
        print(f"{parsed_argument} = {getattr(args, parsed_argument)}")