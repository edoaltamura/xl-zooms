import argparse
from register import calibration_zooms

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument(
    '-k',
    '--keywords',
    type=str,
    nargs='+',
    required=True
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
    choices=list(
        range(
            len(calibration_zooms.get_snap_redshifts())
        )
    )
)

parser.add_argument(
    '-m',
    '--mass-estimator',
    type=str.lower,
    default='true',
    required=True,
    choices=['true', 'hse', 'spec']
)

parser.add_argument(
    '-q',
    '--quiet',
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

if args.quiet:
    mpl_backend = 'Agg'
else:
    mpl_backend = 'TkAgg'
matplotlib.use(mpl_backend)

# Apply the matplotlib stylesheet
import matplotlib.pyplot as plt
try:
    plt.style.use("../mnras.mplstyle")
except:
    pass

