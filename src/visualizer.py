from typing import Optional, Tuple
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import os

mpl.use("Agg")

results_dir: str = "results"


def set_results_dir(new_dir: str) -> None:
    global results_dir
    results_dir = new_dir
    os.makedirs(results_dir, exist_ok=True)


def plot_versus(
    data_x: np.ndarray,
    data_y: np.ndarray,
    name_x: Optional[str] = "X",
    name_y: Optional[str] = "Y",
    units_x: Optional[str] = None,
    units_y: Optional[str] = None,
    trim: Tuple[int, int] = (0, 0),
    lines: Optional[bool] = False,
    colour: Optional[str] = "r",
):
    # trim the starts and end of data
    trim_start, trim_end = trim
    max_size = min(len(data_x), len(data_y))
    data_x = data_x[trim_start : max_size - trim_end]
    data_y = data_y[trim_start : max_size - trim_end]

    # create a figure that is 6in x 6in
    fig = plt.figure()

    # the axis limits and grid lines
    plt.grid(True)

    unit_x_str = f" ({units_x})" if units_x is not None else ""
    unit_y_str = f" ({units_y})" if units_y is not None else ""

    trim_str = " [" + str(trim_start) + ", " + str(trim_end) + "]"

    # label your graph, axes, and ticks on each axis
    plt.xlabel(name_x + unit_x_str, fontsize=16)
    plt.ylabel(name_y + unit_y_str, fontsize=16)
    plt.xticks()
    plt.yticks()
    plt.tick_params(labelsize=15)
    if name_x == "":
        plt.title(name_y + trim_str, fontsize=18)
    else:
        plt.title(name_y + " versus " + name_x + trim_str, fontsize=18)

    # plot dots
    if lines:
        plt.plot(data_x, data_y, color=colour, linewidth=1)
    else:
        plt.plot(data_x, data_y, colour + "o")

    # complete the layout, save figure, and show the figure for you to see
    plt.tight_layout()
    filename = name_y + "_vs_" + name_x + ".png" if name_x != "" else name_y + ".png"
    filename = f"{name_y} x {name_x}.png"
    save_figure_to_file(fig, filename)


def plot_histogram2d(
    data_x: np.ndarray,
    data_y: np.ndarray,
    name_x: Optional[str] = "X",
    name_y: Optional[str] = "Y",
    units_x: Optional[str] = None,
    units_y: Optional[str] = None,
    bins: Optional[int] = 50,
    cmap: Optional[str] = "hot",
):
    fig = plt.figure()
    plt.hist2d(data_x, data_y, bins=bins, cmap=cmap)
    cb = plt.colorbar()
    cb.set_label("Frequency")
    title: str = f"{name_x} x {name_y} Histogram2D"
    plt.title(title)
    unit_x_str = f" ({units_x})" if units_x is not None else ""
    unit_y_str = f" ({units_y})" if units_y is not None else ""
    plt.xlabel(f"{name_x}{unit_x_str}")
    plt.ylabel(f"{name_y}{unit_y_str}")
    plt.tight_layout()
    # save to disk
    filename = f"{title}.png"
    save_figure_to_file(fig, filename)


def plot_diff(
    subA,
    subB,
    units="",
    name_A="A",
    name_B="B",
    trim=(0, 0),
    lines=False,
    colour="r",
    dir_path="results",
):
    # TODO: refactor
    raise NotImplementedError
    # trim the starts and end of data
    trim_start, trim_end = trim
    max_size = min(len(subA), len(subB))
    subA = subA[trim_start : max_size - trim_end]
    subB = subB[trim_start : max_size - trim_end]

    # create a figure that is 6in x 6in
    fig = plt.figure()

    # the axis limits and grid lines
    plt.grid(True)

    units_str = " (" + units + ")" if units != "" else ""
    trim_str = " [" + str(trim_start) + ", " + str(trim_end) + "]"

    # label your graph, axes, and ticks on each axis
    plt.xlabel("Points", fontsize=16)
    plt.ylabel("Difference" + units_str, fontsize=16)
    plt.xticks()
    plt.yticks()
    plt.tick_params(labelsize=15)
    plt.title("Difference (" + name_A + " - " + name_B + ")" + trim_str, fontsize=18)

    # generate data
    x_data = np.arange(len(subA))
    y_data = subA - subB
    plt.plot(x_data, y_data, colour + "o")
    if lines:
        plt.plot(x_data, y_data, color=colour, linewidth=1)

    # complete the layout, save figure, and show the figure for you to see
    plt.tight_layout()

    # make file and save to disk
    if not os.path.exists(os.path.join(os.getcwd(), dir_path)):
        os.mkdir(dir_path)
    filename = name_A + "_minus_" + name_B + ".png"
    fig.savefig(os.path.join(dir_path, filename))
    plt.close(fig)


def save_figure_to_file(
    fig: plt.Figure,
    filename: str,
    dir_path: Optional[str] = None,
    silent: Optional[bool] = False,
) -> None:
    # make file and save to disk
    if dir_path is None:
        global results_dir
        dir_path = results_dir
    if not os.path.exists(os.path.join(os.getcwd(), dir_path)):
        os.mkdir(dir_path)
    filename: str = filename.lower().replace(
        " ", "_"
    )  # all lowercase, use _ instead of spaces
    fig.savefig(os.path.join(dir_path, filename))
    plt.close(fig)
    if not silent:
        print(f"output figure to {dir_path}/{filename}")


def plot_vector_vs_time(
    xyz: np.ndarray,
    t: np.ndarray,
    title: str,
    ax_titles: Optional[Tuple[str]] = ("X", "Y", "Z"),
    silent: Optional[bool] = False,
) -> None:
    n, d = xyz.shape
    assert xyz.shape == (n, d)
    assert t.shape == (n,)
    fig = plt.figure()
    fig.suptitle(title)
    gs = fig.add_gridspec(d, hspace=0)
    axs = gs.subplots(sharex=True, sharey=False)
    for dim in range(d):
        data_dim = xyz[:, dim]
        axs[dim].set(ylabel=ax_titles[dim] if dim < len(ax_titles) else "")
        axs[dim].plot(t, data_dim)
    filename: str = f"{title}.png"
    save_figure_to_file(fig, filename, silent=silent)


def plot_3Dt(
    xyz: np.ndarray,
    t: np.ndarray,
    title: Optional[str] = None,
    axes_titles: Optional[Tuple[str]] = ("X", "Y", "Z"),
    interactive: Optional[bool] = False,
) -> None:
    if interactive:
        try:
            mpl.use("TkAgg")
        except Exception as e:
            print(e)
            return
    n = len(t)
    assert xyz.shape == (n, 3)
    assert t.shape == (n,)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    plot = ax.scatter(x, y, z, c=t)
    cb = plt.colorbar(plot, pad=0.2)
    # cb.set_ticklabels(["start", "end"])
    if title is not None:
        ax.set_title(title)
    assert len(axes_titles) == 3
    ax.set_xlabel(axes_titles[0])
    ax.set_ylabel(axes_titles[1])
    ax.set_zlabel(axes_titles[2])
    if interactive:
        plt.show()
        mpl.use("Agg")  # back to non gui-based
    else:
        filename: str = f"{title}.png"
        save_figure_to_file(fig, filename)
