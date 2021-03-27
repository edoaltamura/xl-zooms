from matplotlib import pyplot as plt
from matplotlib.collections import PathCollection
from pandas import DataFrame

allowed_backends = ['TkAgg', 'Qt5Agg']


def latex_float(f):
    float_str = f"{f:.2g}"
    if f < 1e-2:
        float_str = f"{f:.0e}"
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def interactive_hover(
        scatter_canvas: PathCollection,
        results_catalogue: DataFrame
) -> None:
    if plt.get_backend() not in allowed_backends:
        return

    fig = scatter_canvas.figure
    ax = scatter_canvas.axes

    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(
            boxstyle="round",
            fc="w"
        ),
        arrowprops=dict(arrowstyle="->")
    )
    annot.set_visible(False)

    def update_annotation(ind):
        point_index = ind["ind"][0]
        pos = scatter_canvas.get_offsets()[point_index]
        annot.xy = pos

        text = ''
        for column_name in results_catalogue.columns:
            field_value = results_catalogue.loc[point_index, column_name]
            if not isinstance(field_value, str):
                field_value = latex_float(field_value)
            text += f"{column_name}: {field_value}\n"

        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor(
            scatter_canvas.get_facecolor()[point_index]
        )
        annot.get_bbox_patch().set_alpha(0.3)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = scatter_canvas.contains(event)
            if cont:
                update_annotation(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
