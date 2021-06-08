import matplotlib
import matplotlib.pyplot as plt

from .static_parameters import matplotlib_stylesheet


def set_mpl_backend(is_quiet: bool):
    mpl_backend = 'Agg' if is_quiet else 'TkAgg'
    matplotlib.use(mpl_backend)


def set_mnras_stylesheet():
    try:
        plt.style.use(matplotlib_stylesheet)
    except (FileNotFoundError, OSError, NotADirectoryError):
        print('Could not find the mnras.mplstyle style-sheet.')
