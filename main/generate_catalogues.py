import sys

sys.path.append("..")

from scaling_relations import *
from register import dump_memory_usage

dump_memory_usage()

catalogue_starters = [
    # VRProperties(),
    # GasFractions(),
    # StarFractions(),
    # MWTemperatures(),
    # Relaxation(),
    # XrayLuminosities(),
    # Entropies(),
    SphericalOverdensities(density_contrast=1500.),
]

dump_memory_usage()

for cs in catalogue_starters:
    print(f"[Analysis] Computing {type(cs).__name__:s}...")

    try:
        cs.process_catalogue()
    except Exception as e:
        print(e, "Continue to next iteration.", sep='\n')
        pass

    dump_memory_usage()
