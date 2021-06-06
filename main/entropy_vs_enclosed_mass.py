import sys
from matplotlib import pyplot as plt

sys.path.append("..")

from scaling_relations import EntropyFgasSpace
from register import find_files, set_mnras_stylesheet, xlargs


snap, cat = find_files()

# set_mnras_stylesheet()
profile_obj = EntropyFgasSpace(max_radius_r500=1.5)
profile_obj.display_single_halo(path_to_snap=snap, path_to_catalogue=cat)

# print(
#     gf.process_catalogue()
# )
