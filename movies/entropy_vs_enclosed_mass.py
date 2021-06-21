import sys
from matplotlib import pyplot as plt
import traceback


sys.path.append("..")

from scaling_relations import EntropyFgasSpace, EntropyProfiles
from register import find_files, set_mnras_stylesheet, xlargs

snap, cat = find_files()

# set_mnras_stylesheet()

try:

    profile_obj = EntropyFgasSpace(max_radius_r500=1.)

    fig = plt.figure(figsize=(5, 5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.7)
    axes = gs.subplots()


    profile_obj.display_single_halo(path_to_snap=snap, path_to_catalogue=cat)

except Exception as e:
    print(f"Snap number {xlargs.snapshot_number:04d} could not be processed.", e, sep='\n')
    traceback.print_exc()