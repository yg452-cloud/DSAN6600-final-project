from pathlib import Path
import matplotlib.pyplot as plt


# Project-level plotting constants
# These constants make it easier to keep the visual style consistent
DEFAULT_FIGSIZE = (8, 5)
WIDE_FIGSIZE = (10, 5)
DEFAULT_DPI = 300

TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 10

GRID_ALPHA = 0.3
GRID_LINESTYLE = "--"

# Fixed colors for model names
# Keep these consistent across all notebooks and figures
MODEL_COLORS = {
    "unet": "#1f77b4",
    "deeplabv3plus": "#ff7f0e",
    "segformer": "#2ca02c",
}


def set_plot_style():
    """
    Apply a consistent matplotlib style for the whole project.

    This function should usually be called once near the beginning
    of a notebook before creating any figures.
    """
    plt.rcParams["figure.figsize"] = DEFAULT_FIGSIZE
    plt.rcParams["figure.dpi"] = DEFAULT_DPI

    plt.rcParams["axes.titlesize"] = TITLE_SIZE
    plt.rcParams["axes.labelsize"] = LABEL_SIZE

    plt.rcParams["xtick.labelsize"] = TICK_SIZE
    plt.rcParams["ytick.labelsize"] = TICK_SIZE

    plt.rcParams["legend.fontsize"] = LEGEND_SIZE

    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = GRID_ALPHA
    plt.rcParams["grid.linestyle"] = GRID_LINESTYLE

    plt.rcParams["savefig.dpi"] = DEFAULT_DPI
    plt.rcParams["savefig.bbox"] = "tight"


def save_figure(fig, save_path, dpi=DEFAULT_DPI, close_fig=False):
    """
    Save a matplotlib figure with consistent export settings.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    save_path : str or Path
        Output file path.
    dpi : int, optional
        Export resolution. Default is DEFAULT_DPI.
    close_fig : bool, optional
        Whether to close the figure after saving.
        Default is False.

    Notes
    -----
    This function automatically creates parent directories if needed.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if close_fig:
        plt.close(fig)