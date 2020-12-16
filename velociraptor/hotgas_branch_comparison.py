import os

os.chdir("/cosma/home/dp004/dc-alta2/data7/xl-zooms/hydro")

runs = [
    "L0300N0564_VR121_-8res_Ref",
    "L0300N0564_VR1236_-8res_Ref",
    "L0300N0564_VR130_-8res_Ref",
    "L0300N0564_VR139_-8res_Ref",
    "L0300N0564_VR155_-8res_Ref",
    "L0300N0564_VR187_-8res_Ref",
    "L0300N0564_VR18_-8res_Ref",
    "L0300N0564_VR2272_-8res_Ref",
    "L0300N0564_VR23_-8res_Ref",
    "L0300N0564_VR2414_-8res_Ref",
    "L0300N0564_VR2766_-8res_Ref",
    "L0300N0564_VR2905_-8res_Ref",
    "L0300N0564_VR2915_-8res_Ref",
    "L0300N0564_VR3032_-8res_Ref",
    "L0300N0564_VR340_-8res_Ref",
    "L0300N0564_VR36_-8res_Ref",
    "L0300N0564_VR37_-8res_Ref",
    "L0300N0564_VR470_-8res_Ref",
    "L0300N0564_VR485_-8res_Ref",
    "L0300N0564_VR55_-8res_Ref",
    "L0300N0564_VR666_-8res_Ref",
    "L0300N0564_VR680_-8res_Ref",
    "L0300N0564_VR775_-8res_Ref",
    "L0300N0564_VR801_-8res_Ref",
    "L0300N0564_VR813_-8res_Ref",
    "L0300N0564_VR918_-8res_Ref",
    "L0300N0564_VR93_-8res_Ref",
]

for run in runs:
    os.system(f"export OMP_NUM_THREADS=28;VELOCIraptor-STF_hotgas_2020/stf -I 2 -C vrconfig_3dfofbound_subhalos_SO_hydro.cfg -i {run}/snapshots/{run}_2749 -o Ref_VR_comparison_analysis/{run}")