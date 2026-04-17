import matplotlib.pyplot as plt


FIGSIZE = (8.0, 4.8)
DPI = 300

ALGO_COLORS = {
    "QL": "#1f77b4",
    "PPO": "#d55e00",
    "Fixed-Time": "#2a9d8f",
}

NEUTRAL_COLORS = {
    "raw": "#c7c7c7",
    "smooth": "#0b3c5d",
}

METRIC_LABELS = {
    "mean_wait": "Mean Waiting Time (s)",
    "total_wait": "Total Waiting Time (s)",
    "mean_speed": "Mean Speed (m/s)",
    "total_stopped": "Stopped Vehicles",
    "stopped_ratio": "Stopped Ratio",
    "arrived_last": "Arrived Vehicles",
    "mean_waiting_time": "Mean Waiting Time (s)",
}


def apply_publication_style():
    plt.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "figure.dpi": DPI,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#dddddd",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
        }
    )


def style_axis(ax, title, xlabel="Episode", ylabel=None):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def finish_and_save(fig, save_path):
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
