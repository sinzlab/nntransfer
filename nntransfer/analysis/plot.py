import os
import string

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from nnfabrik.utility.dj_helpers import make_hash


def plot(plot_function):
    def plot_wrapper(*args, **kwargs):
        nrows = kwargs.pop("nrows", 1)
        ncols = kwargs.pop("ncols", 1)
        gridspec_kw = kwargs.pop("gridspec_kw", None)
        style = kwargs.pop("style", "lighttalk")
        legend_outside = kwargs.pop("legend_outside", False)
        title = kwargs.pop("title", "")
        x_label = kwargs.pop("x_label", "")
        y_label = kwargs.pop("y_label", "")
        rotate_x_labels = kwargs.pop("rotate_x_labels", False)
        despine = kwargs.pop("despine", True)
        panel_labels = kwargs.pop("panel_labels", False)
        save = kwargs.pop("save", "")
        figure_heigh_scale = kwargs.pop("figure_height_scale", 1.0)
        tight = kwargs.pop("tight", False)


        fig = kwargs.pop("fig", None)
        ax = kwargs.pop("ax", None)

        if fig and ax:
            plot_function(*args, fig=fig, ax=ax, **kwargs)
            return fig, ax

        fs = set_size(
            style=style,
            ratio=kwargs.pop("ratio", None),
            fraction=kwargs.pop("fraction", 1.0),
            subplots=(ncols, nrows),
            gridspec_kw=gridspec_kw,
        )
        fs = (fs[0], fs[1] * nrows / (ncols) * figure_heigh_scale)
        sns.set()

        if "light" in style:
            sns.set_style("whitegrid")
        if "ticks" in style:
            sns.set_style("ticks")
        if "dark" in style:
            plt.style.use("dark_background")
        if "talk" in style or "beamer" in style:
            sns.set_context("talk")
            font = "sans-serif"
            small_fontsize = 13
            xsmall_fontsize = 13
            normal_fontsize = 14
            large_fontsize = 17
        else:
            sns.set_context("paper")
            font = "serif"
            small_fontsize = 8
            xsmall_fontsize = 7
            normal_fontsize = 10
            large_fontsize = 12
        sns.set_style(
            "whitegrid",
            {
                "axes.edgecolor": "0.1",
                "xtick.bottom": True,
                "xtick.top": False,
                "ytick.left": True,
                "ytick.right": False,
            },
        )
        nice_fonts = {
            "font.family": font,
            "axes.labelsize": normal_fontsize,
            "font.size": normal_fontsize,
            "legend.fontsize": xsmall_fontsize,
            "xtick.labelsize": small_fontsize,
            "ytick.labelsize": small_fontsize,
        }
        if "tex" in style:
            nice_fonts["text.usetex"] = True
            nice_fonts["pgf.rcfonts"] = False
            nice_fonts["pgf.texsystem"] = "pdflatex"
            mpl.rcParams['axes.unicode_minus'] = False
            mpl.use("pgf")
        mpl.rcParams.update(nice_fonts)

        fig, ax = plt.subplots(
            nrows,
            ncols,
            figsize=fs,
            dpi=kwargs.pop("dpi", None),
            sharex=kwargs.pop("sharex", False),
            sharey=kwargs.pop("sharey", False),
            gridspec_kw=gridspec_kw,
        )

        if nrows == 1:
            ax = [ax]
        if ncols == 1:
            for i in range(len(ax)):
                ax[i] = [ax[i]]

        plt.grid(True, linestyle=":", linewidth=0.5)

        # execute the actual plotting
        plot_function(*args, fig=fig, ax=ax, **kwargs)

        if panel_labels:
            for i in range(len(ax)):
                for j in range(len(ax[i])):
                    label = string.ascii_uppercase[i * len(ax[i]) + j]
                    ax[i][j].text(
                        0.04,
                        1.00,
                        label,
                        transform=ax[i][j].transAxes,
                        fontsize=normal_fontsize,
                        fontweight="bold",
                        va="top",
                    )
        if title:
            fig.suptitle(title, fontsize=large_fontsize)
        if y_label:
            plt.ylabel(y_label, fontsize=normal_fontsize)
        if x_label:
            plt.xlabel(x_label, fontsize=normal_fontsize)
        if tight:
            fig.tight_layout()
        if rotate_x_labels:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        if despine:
            sns.despine(offset=3, trim=False)
        legend_args = {
            "fontsize": xsmall_fontsize,
            "title_fontsize": f"{small_fontsize}",
            "frameon": False,
            "borderaxespad": 0.0,
            "loc": "best",
        }
        if legend_outside:
            legend_args.update({"bbox_to_anchor": (1.05, 1), "loc": 2})
        plt.legend(**legend_args)

        if save:
            save_plot(
                fig,
                save + "_" + style,
                types=("png", "pdf", "pgf") if "tex" in style else ("png", "pdf"),
            )

        return fig, ax

    return plot_wrapper


def set_size(style, ratio=None, fraction=1, subplots=(1, 1), gridspec_kw=None):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    style: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if "thesis" in style:
        width_pt = 426.79135
    elif "beamer" in style:
        width_pt = 398.3386
    elif "pnas" in style:
        width_pt = 246.09686
    elif "nips" in style or "iclr" in style:
        width_pt = 397.48499  # \usepackage{layouts} ... \printinunitsof{pt}\prntlen{\textwidth}
    elif "paper" in style or "talk" in style:
        width_pt = 1000
    else:
        width_pt = style

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    if ratio:
        ratio = ratio[0] / ratio[1]
    else:
        # Golden ratio to set aesthetic figure height
        # https://disq.us/p/2940ij3
        ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if gridspec_kw and "height_ratios" in gridspec_kw:
        fig_height_in = 0
        max_height = max(gridspec_kw["height_ratios"])
        for h_ratio in gridspec_kw["height_ratios"]:
            fig_height_in += fig_width_in * ratio * (h_ratio / max_height)
    else:
        fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)


def save_plot(fig, name, types=("pgf", "pdf", "png")):
    for file_type in types:
        fig.savefig(
            name + "." + file_type,
            facecolor=fig.get_facecolor(),
            edgecolor=fig.get_edgecolor(),
            bbox_inches="tight",
        )
    plt.close(fig)


def generate_gif(filenames, out_name):
    import imageio

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(
        "{}.gif".format(out_name),
        images,
        format="GIF",
        duration=2,
    )
