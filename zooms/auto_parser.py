import argparse
import matplotlib.pyplot as plt
from register import calibration_zooms

parser = argparse.ArgumentParser()

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

args = parser.parse_args()


try:
    plt.style.use("../mnras.mplstyle")
except:
    pass
