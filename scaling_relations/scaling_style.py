import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap

name = "Set2"
cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
colors = cmap.colors  # type: list

style_subgrid_model = pd.DataFrame(
    [
        ['fixedAGNdT7.5_Nheat1_SNnobirth', colors[0], '.', 14, '-', 1,
         Patch(facecolor=colors[0], edgecolor='None', label=r'fixed AGN $dT=10^{7.5}$ K')],
        ['fixedAGNdT8_Nheat1_SNnobirth', colors[1], '.', 14, '-', 1,
         Patch(facecolor=colors[1], edgecolor='None', label=r'fixed AGN $dT=10^8$ K')],
        ['fixedAGNdT8.5_Nheat1_SNnobirth', colors[2], '.', 14, '-', 1,
         Patch(facecolor=colors[2], edgecolor='None', label=r'fixed AGN $dT=10^{8.5}$ K')],
        ['fixedAGNdT9_Nheat1_SNnobirth', colors[3], '.', 14, '-', 1,
         Patch(facecolor=colors[3], edgecolor='None', label=r'fixed AGN $dT=10^9$ K')],
        ['fixedAGNdT9.5_Nheat1_SNnobirth', colors[4], '.', 14, '-', 1,
         Patch(facecolor=colors[4], edgecolor='None', label=r'fixed AGN $dT=10^{9.5}$ K')],
        ['Ref', colors[5], '.', 14, '-', 1,
         Patch(facecolor=colors[5], edgecolor='None', label=r'Random AGN fixed $dT$')],
        ['MinimumDistance', colors[6], '.', 14, '-', 1,
         Patch(facecolor=colors[6], edgecolor='None', label=r'Minimum distance AGN fixed $dT$')],
        ['Isotropic', colors[7], '.', 14, '-', 1,
         Patch(facecolor=colors[7], edgecolor='None', label=r'Isotropic AGN fixed $dT$')],
    ],
    columns=['Run name keyword', 'Color', 'Marker style', 'Marker size', 'Line style', 'Line width', 'Legend handle']
)


def get_style_for_object(run_name: str):
    for i in len(style_subgrid_model):
        if run_name.endswith(style_subgrid_model.loc[i, 'Run name keyword']):
            return style_subgrid_model.loc[i]
