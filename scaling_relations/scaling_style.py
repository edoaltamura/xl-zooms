import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap

name = "Spectral"
cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
# colors = cmap.colors  # type: list
colors = cmap(np.linspace(0, 255, 8, dtype=np.int))

style_subgrid_model = pd.DataFrame(
    [
        ['fixedAGNdT7.5_Nheat1_SNnobirth', colors[0], '.', 23, '-', 1,
         Patch(facecolor=colors[0], edgecolor='None', label=r'fixed AGN $dT=10^{7.5}$ K')],
        ['fixedAGNdT8_Nheat1_SNnobirth', colors[1], '.', 23, '-', 1,
         Patch(facecolor=colors[1], edgecolor='None', label=r'fixed AGN $dT=10^8$ K')],
        ['fixedAGNdT8.5_Nheat1_SNnobirth', colors[2], '.', 23, '-', 1,
         Patch(facecolor=colors[2], edgecolor='None', label=r'fixed AGN $dT=10^{8.5}$ K')],
        ['fixedAGNdT9_Nheat1_SNnobirth', colors[3], '.', 23, '-', 1,
         Patch(facecolor=colors[3], edgecolor='None', label=r'fixed AGN $dT=10^9$ K')],
        ['fixedAGNdT9.5_Nheat1_SNnobirth', colors[4], '.', 23, '-', 1,
         Patch(facecolor=colors[4], edgecolor='None', label=r'fixed AGN $dT=10^{9.5}$ K')],
        ['Ref', colors[5], '.', 23, '-', 1,
         Patch(facecolor=colors[5], edgecolor='None', label=r'Random AGN variable $dT$')],
        ['MinimumDistance', colors[6], '.', 23, '-', 1,
         Patch(facecolor=colors[6], edgecolor='None', label=r'Minimum distance AGN variable $dT$')],
        ['Isotropic', colors[7], '.', 23, '-', 1,
         Patch(facecolor=colors[7], edgecolor='None', label=r'Isotropic AGN variable $dT$')],
    ],
    columns=['Run name keyword', 'Color', 'Marker style', 'Marker size', 'Line style', 'Line width', 'Legend handle']
)


def get_style_for_object(run_name: str):
    for index, row in style_subgrid_model.iterrows():
        if run_name.endswith(row['Run name keyword']):
            return row
