import pandas as pd
from matplotlib.cm import get_cmap

name = "Set2"
cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
colors = cmap.colors  # type: list

style_subgrid_model = pd.DataFrame(
    [
        ['fixedAGNdT7.5', colors[0], '.', 14, '-', 1],
        ['fixedAGNdT8', colors[1], '.', 14, '-', 1],
        ['fixedAGNdT8.5', colors[2], '.', 14, '-', 1],
        ['fixedAGNdT9', colors[3], '.', 14, '-', 1],
        ['fixedAGNdT9.5', colors[4], '.', 14, '-', 1],
    ],
    columns=['Run name keyword', 'Color', 'Marker style', 'Marker size', 'Line style', 'Line width']
)
