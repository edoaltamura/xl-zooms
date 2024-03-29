import h5py

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
# for run in runs:
#     with h5py.File(f'Ref_VR_comparison_analysis_noOMP/{run}.properties', 'r') as h5file:
#         print(h5file['/SO_Mass_500_rhocrit'][0], h5file['/SO_Mass_gas_highT_1.000000_times_500.000000_rhocrit'][0])
#
# def running(run: str):
#     import os
#     os.system(f"VELOCIraptor-STF_hotgas_2020/stf -I 2 -C vrconfig_3dfofbound_subhalos_SO_hydro.cfg -i {run}/snapshots/{run}_2749 -o Ref_VR_comparison_analysis_noOMP/{run}")
#
#
# from multiprocessing import Pool, cpu_count
#
# with Pool() as pool:
#     print(f"Analysis mapped onto {cpu_count():d} CPUs.")
#     results = pool.map(running, iter(runs))



import numpy as np

hotgas = np.array([
    [10470.871305061322, 1205.65234375],
    [1927.8919169139967, 129.640380859375],
    [9872.485173727897, 1084.92041015625],
    [8633.183958648162, 950.6312255859375],
    [9291.25696547179, 1095.5577392578125],
    [8828.982009168481, 914.2821044921875],
    [29078.73349020424, 3587.639892578125],
    [1108.3664271757536, 47.246437072753906],
    [24184.3916103626, 3064.78173828125],
    [983.3178222680216, 46.0886344909668],
    [941.9762313801245, 38.413673400878906],
    [923.5887049489812, 43.418548583984375],
    [832.4801160104131, 33.174781799316406],
    [755.9020052795005, 25.031953811645508],
    [5047.376606709369, 425.9169616699219],
    [20805.08007116237, 2755.0859375],
    [23177.980938143413, 3046.862548828125],
    [4353.688247323345, 336.8359069824219],
    [4114.599545566402, 352.1247253417969],
    [17266.19556747952, 2029.6187744140625],
    [3493.1009422858424, 257.8345947265625],
    [3373.0609190198948, 274.95721435546875],
    [3108.1507966398594, 175.1416473388672],
    [2644.130269874521, 194.98504638671875],
    [2958.4123721371257, 266.6622009277344],
    [2670.1405651982373, 184.58700561523438],
    [12785.215476010579, 1493.7310791015625]
]).T * 1e10

# Data structure: m500c | M_hotgas500c
master = np.array([
    [10470.778938888074, 1208.31606784],
    [1927.8469759419206, 129.6077684736],
    [9871.424285268791, 1087.3213026304],
    [8632.392450139316, 952.2142773248],
    [9289.475433633304, 1097.5129370624],
    [8830.894004975302, 916.2599694336],
    [29076.002248371067, 3610.5252503552],
    [1108.4000225736427, 47.2563449856],
    [24186.37856320545, 3084.1098993664],
    [983.2513122823614, 46.097006592],
    [941.6456506829347, 38.407471104],
    [923.1080286108768, 43.471781888],
    [832.263188200671, 33.1603247104],
    [756.1849241016072, 25.0338541568],
    [5049.713203476258, 426.8838617088],
    [20804.646156370913, 2770.228740096],
    [23173.333138295755, 3063.8367309824],
    [4346.374498199273, 336.9870819328],
    [4115.264417465431, 352.4420435968],
    [17268.02247236218, 2038.4493600768],
    [3492.9856775677013, 257.9752812544],
    [3372.7856002512513, 275.0908465152],
    [3107.3832307154908, 175.146467328],
    [2644.488432050909, 195.0517428224],
    [2958.403958832529, 266.782244864],
    [2669.971660487996, 184.6446784512],
    [12785.413582786045, 1498.6474684416],
]).T * 1e10

# import matplotlib.pyplot as plt
#
# fig, ax = plt.subplots(figsize=(3.321 * 2, 3.0 * 2))
# ax.scatter(hotgas[0], hotgas[1], marker='.', s=70, c='k', label='hotgas_branch')
# ax.scatter(master[0], master[1], marker='.', s=200, facecolors='none', edgecolors='r', label='master')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlabel(r'$M_{500{\rm crit}}\ [{\rm M}_{\odot}]$')
# ax.set_ylabel(r'$M_{{\rm gas},500{\rm crit}}\ [{\rm M}_{\odot}]$')
# ax.plot(ax.get_xlim(), [lim * 0.15741 for lim in ax.get_xlim()], '--', color='k')
#
# plt.legend()
# plt.show()
# fig.savefig('test.png', dpi=400)
