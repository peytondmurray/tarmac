from collections.abc import Sequence

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def _label_offset(ax: plt.Axes, axis: str = "y") -> None:
    """Move ticklabel offsets (e.g. exponents) to the axis label.

    The label is dynamically updated when the axis range changes.

    Parameters
    ----------
    ax : plt.Axes
        Axes instance to label the offset on
    axis : str
        Axis to label of the offset on; can be 'x' or 'y'

    """
    if axis == "y":
        fmt = ax.yaxis.get_major_formatter()
        ax.yaxis.offsetText.set_visible(False)
        labelfunc = ax.set_ylabel
        label = ax.get_ylabel()

    elif axis == "x":
        fmt = ax.xaxis.get_major_formatter()
        ax.xaxis.offsetText.set_visible(False)
        labelfunc = ax.set_xlabel
        label = ax.get_xlabel()

    def update_label(_ax: plt.Axes) -> None:
        offset = fmt.get_offset()
        if offset == "":
            labelfunc(f"{label}")
        else:
            labelfunc(f"{label} ({offset})")

        # Only display labels on axes which lie at the edge of the subplot grid
        ax.label_outer()

    ax.callbacks.connect("ylim_changed", update_label)
    ax.callbacks.connect("xlim_changed", update_label)
    ax.figure.canvas.draw()
    update_label(None)


def corner_plot(
    fig: plt.Figure,
    samples: np.ndarray,
    bins: int | Sequence[int] = 100,
    ranges: Sequence[tuple[float, float]] | None = None,
    labels: Sequence[str] | None = None,
    cmap: str | colors.Colormap = "viridis",
    plot_type: str = "hist",
    facecolor: str = "C0",
    edgecolor: str | None = None,
    density: bool = True,
) -> None:
    """Generate a corner plot.

    Using the provided MCMC samples, generate a corner plot - a set of 2D
    histograms showing the bivariate distributions for each pair of model
    parameters.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure in which to draw the corner plot; should be empty
    samples : np.ndarray
        MCMC samples of shape (nsamples, nwalkers, ndim).
    bins : int | Sequence[int]
        Number of bins along each axis of each histogram.
    ranges : Iterable[tuple[float, float]] | None
        A list of bounds (min, max) for each histogram plot. (the default
        is None, which automatically chooses 3*sigma bounds about the mean.)
    labels : Iterable[str] | None
        List of names of model parameters. Must be of length *ndim* (the
        default is None, which makes blank labels).
    cmap : str | colors.Colormap
        Colormap to use to plot the 2D histograms
    plot_type : str
        Specify the plot type. Should be one of ['hex', 'hist']
    facecolor : str
        Facecolor to use for the 1D histograms
    edgecolor : str | None
        Edgecolor to use for the 1D histograms
    density : bool
        If True, the density is plotted
    """
    if len(np.shape(samples)) != 3:
        raise ValueError(
            f"Samples must be of shape (nsamples, nwalkers, ndim), not {np.shape(samples)}"
        )

    nsamples, _nwalkers, ndim = np.shape(samples)
    samples = samples.reshape((-1, ndim))

    if nsamples <= ndim:
        raise ValueError(
            "Number of samples <= number of dimensions. Is this intended for this dataset?"
        )

    if isinstance(bins, int):
        bins_arr = np.array([bins for _ in range(ndim)])
    else:
        raise ValueError(f"Invalid type {type(bins)} for parameter 'bins'.")

    if ranges is None:
        ranges = [_nice_bounds(samples[:, i]) for i in range(ndim)]
    elif len(ranges) != ndim:
        raise ValueError(
            "Dimension mismatch between ranges and number of columns in samples."
        )
    else:
        ranges = [
            _nice_bounds(samples[:, i]) if ranges[i] is None else ranges[i]
            for i in range(ndim)
        ]

    if labels is None:
        labels = ["" for _ in range(ndim)]

    # Divide the figure into a bunch of subplots, and Remove whitespace
    # between plots
    axes = fig.subplots(ndim, ndim, sharex="col")
    fig.subplots_adjust(
        left=0.1, bottom=0.1, right=0.98, top=0.98, wspace=0.05, hspace=0.05
    )

    if ndim == 1:
        _hist_1d(
            ax=axes,
            samples=samples[:, 0],
            bins=bins_arr[0],
            bounds=ranges[0],
            label=labels[0],
            density=density,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )

    else:
        for i in range(ndim):
            # Plot the 1D histograms along the diagonal. If i == ndim-1,
            # make xticklabels. Otherwise, omit them.
            _hist_1d(
                ax=axes[i, i],
                samples=samples[:, i],
                bins=bins_arr[i],
                bounds=ranges[i],
                label=labels[i],
                facecolor=facecolor,
                edgecolor=edgecolor,
            )

            # Plot the 2D histograms in the lower left corner
            for j in range(ndim):
                if j > i:
                    axes[i, j].axis("off")
                elif j < i:
                    _hist_2d(
                        ax=axes[i, j],
                        xsamples=samples[:, j],
                        ysamples=samples[:, i],
                        xbins=bins_arr[j],
                        ybins=bins_arr[i],
                        xbounds=ranges[j],
                        ybounds=ranges[i],
                        xlabel=labels[j],
                        ylabel=labels[i],
                        cmap=cmap,
                        plot_type=plot_type,
                        density=density,
                        sharey=axes[i, 0],
                    )

                for tick in axes[i, j].get_xticklabels():
                    tick.set_rotation(45)


def _hist_1d(
    ax: plt.Axes,
    samples: np.ndarray,
    bins: int,
    bounds: tuple[float, float],
    label: str,
    density: bool = True,
    facecolor: str = "C0",
    edgecolor: str | None = None,
) -> None:
    pdf, xedges = np.histogram(samples, bins=bins, range=bounds, density=density)
    pdf = np.append(pdf, 0)
    ax.fill_between(xedges, pdf, step="post", facecolor=facecolor, edgecolor=edgecolor)

    ax.set_yticklabels([])
    ax.set_xlim(_nice_bounds(samples))

    ax.set_xlabel(label)
    ax.get_xaxis().set_major_locator(ticker.MaxNLocator(nbins=5, prune="upper"))
    _label_offset(ax, "x")


def _hist_2d(
    ax: plt.Axes,
    xsamples: np.ndarray,
    ysamples: np.ndarray,
    xbins: np.ndarray,
    ybins: np.ndarray,
    xbounds: tuple[float, float],
    ybounds: tuple[float, float],
    xlabel: str,
    ylabel: str,
    cmap: str | colors.Colormap,
    plot_type: str,
    density: bool = True,
    sharey: plt.Axes | None = None,
) -> None:
    if plot_type in ["hist", "histogram"]:
        # matplotlib's ax.hist2d makes a patch for each bin (bug?). Instead, use imshow to make a cleaner, faster plot.
        #
        # By default, np.histogram2d histograms x-values along the first dimension of the pdf, and y-values along
        # the second dimension. This is opposite to how we want to display the data, which is why the x and y values
        # are swapped here.
        pdf, yedges, xedges = np.histogram2d(
            ysamples,
            xsamples,
            bins=[ybins, xbins],
            range=[ybounds, xbounds],
            density=density,
        )

        ax.imshow(
            pdf,
            extent=[xbounds[0], xbounds[1], ybounds[0], ybounds[1]],
            cmap=cmap,
            interpolation="nearest",
            origin="lower",
        )
    elif plot_type in ["hex", "hexbin"]:
        ax.hexbin(
            xsamples,
            ysamples,
            gridsize=[int(0.5 * xbins), int(0.5 * ybins)],
            extent=[*xbounds, *ybounds],
            cmap=cmap,
        )
    else:
        raise ValueError(f"Invalid plot_type: {plot_type}")

    ax.set_aspect("auto")

    if sharey is not None:
        ax.sharey(sharey)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _label_offset(ax, "x")
    _label_offset(ax, "y")

    ax.get_xaxis().set_major_locator(ticker.MaxNLocator(nbins=5, prune="upper"))
    ax.get_yaxis().set_major_locator(ticker.MaxNLocator(nbins=5, prune="upper"))
    ax.set_xlim(_nice_bounds(xsamples))
    ax.set_ylim(_nice_bounds(ysamples))

    # Only display labels on axes which lie at the edge of the subplot grid
    ax.label_outer()

    return


def _nice_bounds(samplesx: np.ndarray, factor: float = 3) -> tuple[float, float]:
    """Generate sensible limits for distribution plots.

    Finds the mean+factor*std_dev and mean-factor*std_dev of a set of samples.

    Parameters
    ----------
    samplesx : {ndarray}
        Samples from a distribution.
    factor : {int}, optional
        Number of standard deviations to include (the default is 3, which
        usually gives nice looking plots without being too zoomed out)

    Returns
    -------
    tuple
        (lower limit, upper limit) of plot.
    """
    sx = factor * np.std(samplesx)
    avgx = np.mean(samplesx)
    return avgx - sx, avgx + sx


def walker_trace(
    fig: plt.Figure,
    samples: np.ndarray,
    labels: Sequence[str | None] | None = None,
    **kwargs: str | float,
) -> None:
    """Generate a walker trace figure from MCMC samples.

    Given some input MCMC samples, generate a figure with ndim subplots, one
    for each model parameter, showing the traces of each walker through the
    parameter subspace.

    Parameters
    ----------
    fig : plt.Figure
        Empty Matplotlib figure in which to draw the walker trace subplots.
    samples : np.ndarray
        Output of MCMC sampler, must be of shape (nsamples, nwalkers, ndim).
    labels : Sequence[str] | None
        List of length `ndim` containing variable names for each parameter.
        (the default is None, which means your parameters are unlabeled.)
    **kwargs : dict
        Line2D properties to pass to matplotlib
    """
    nsteps, nwalkers, ndim = np.shape(samples)

    if labels is None:
        labels = [None for _ in range(ndim)]

    if "color" not in kwargs:
        kwargs["color"] = "k"
    if "alpha" not in kwargs:
        kwargs["alpha"] = 0.3

    axes = fig.subplots(ndim, 1)
    fig.subplots_adjust(
        left=0.1, bottom=0.1, right=0.98, top=0.98, wspace=0.05, hspace=0.05
    )

    if ndim == 1:
        axes.plot(samples[:, :, 0], **kwargs)

        axes.set_xlim(0, nsteps)
        if labels[0] is not None:
            axes.set_ylabel(labels[0])
        _label_offset(axes, "y")
        axes.set_xlabel("Step")

    else:
        for i in range(ndim):
            axes[i].plot(samples[:, :, i], **kwargs)

            if i < ndim - 1:
                axes[i].set_xticklabels([])

            axes[i].set_xlim(0, nsteps)
            if labels[i] is not None:
                axes[i].set_ylabel(labels[i])

            _label_offset(axes[i], "y")

        axes[ndim - 1].set_xlabel("Step")

    return
