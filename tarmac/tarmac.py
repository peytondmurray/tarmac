import numpy as np
import matplotlib.ticker as ticker

def labelOffset(ax, axis="y"):
	"""

	Removes axis ticklabel offsets (e.g. exponents) and moves them to the axis label. Label is dynamically updated
	when axis range changes.

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

	def updateLabel(event_axes):
		offset = fmt.get_offset()
		if offset == '':
			labelfunc("{}".format(label))
		else:
			labelfunc("{} ({})".format(label, offset))
		return

	ax.callbacks.connect("ylim_changed", updateLabel)
	ax.callbacks.connect("xlim_changed", updateLabel)
	ax.figure.canvas.draw()
	updateLabel(None)
	return

def corner_plot(fig, samples, bins=100, ranges=None, labels=None, cmap='viridis', plotType='hist'):
	"""Generate a corner plot.
	
	Using MCMC samples, generate a corner plot - a set of 2D histograms showing the bivariate distributions for each pair of model parameters.
	
	Parameters:
	----------
	fig : {matplotlib.figure.Figure}
		Matplotlib figure in which to draw the corner plot. Should be empty.
	samples : {numpy.ndarray}
		MCMC samples of schape (nwalkers, nsamples, ndim).
	bins : {int}, optional
		Number of bins along each axis of each histogram.
	ranges : {sequence}, optional
		A list of bounds (min, max) for each histogram plot. (the default is None, which automatically chooses 3*sigma bounds about the mean.)
	labels : {list}, optional
		List of names of model parameters. Must be of length *ndim* (the default is None, which makes blank labels).
	cmap : {[type]}, optional
		[description] (the default is None, which [default_description])
	plotType : {str}, optional
		Specify the plot type. Should be one of
		* 'hex'
		* 'hist'

	"""

	#Handling the arguments
	if len(np.shape(samples)) != 3:
		raise ValueError("Samples must be of shape (*nwalkers, nsamples, ndim*), but is of shape {}".format(np.shape(samples)))
	else:
		_, nsamples, ndim = np.shape(samples)
		samples = samples.reshape((-1, ndim))

		if nsamples <= ndim:
			raise ValueError("Number of samples <= number of dimensions. Is this really what you want for this dataset?")

	if isinstance(bins, int):
		bins = np.array([bins for _ in range(ndim)])
	elif len(np.shape(bins)) != 1:
		raise ValueError("Bins should be a 1D array or an integer.")
	elif np.shape(bins)[0] != ndim:
		raise ValueError("Dimension mismatch between bins and number of parameters in samples.")
	else:
		raise ValueError("Invalid type {} for parameter 'bins'.".format(type(bins)))

	if ranges is None:
		ranges = [nice_bounds(samples[:,i]) for i in range(ndim)]
	elif len(ranges) != ndim:
		raise ValueError("Dimension mismatch between ranges and number of columns in samples.")
	else:
		ranges = [nice_bounds(samples[:,i]) if ranges[i] is None else ranges[i] for i in range(ndim)]

	if labels is None:
		labels = ["" for _ in range(ndim)]

	#Divide the figure into a bunch of subplots, and Remove whitespace between plots
	axes = fig.subplots(ndim, ndim)
	fig.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.98, wspace=0.05, hspace=0.05)

	for i in range(ndim):

		#Plot the 1D histograms along the diagonal. If i == ndim-1, make xticklabels. Otherwise, omit them.
		hist_1d(ax=axes[i,i], samples=samples[:,i], bins=bins[i], bounds=ranges[i], label=labels[i], show_xticklabels=(i == ndim - 1))

		#Plot the 2D histograms in the lower left corner
		for j in range(ndim):

			if j > i:
				axes[i,j].axis('off')
			elif j < i:
				if plotType in ["hist", "histogram"]:
					axes[i,j].hist2d(samples[:,j], samples[:,i], bins=[bins[j], bins[i]], range=[ranges[j], ranges[i]], cmap=cmap)
				elif plotType in ["hex", "hexbin"]:
					axes[i,j].hexbin(samples[:,j], samples[:,i], gridsize=[int(0.5*bins[j]), int(0.5*bins[i])], extent=[*ranges[j], *ranges[i]], cmap=cmap)
				if j == 0:
					axes[i,j].set_ylabel(labels[i])
					labelOffset(axes[i,j], "y")
				else:
					axes[i,j].set_yticklabels([])
				if i == ndim - 1:
					axes[i,j].set_xlabel(labels[j])
					labelOffset(axes[i,j], "x")
				else:
					axes[i,j].set_xticklabels([])

				axes[i,j].get_xaxis().set_major_locator(ticker.MaxNLocator(nbins=5, prune='upper'))
				axes[i,j].get_yaxis().set_major_locator(ticker.MaxNLocator(nbins=5, prune='upper'))
				xlim, ylim = nice_bounds(samples[:,j]), nice_bounds(samples[:,i])
				axes[i,j].set_xlim(xlim)
				axes[i,j].set_ylim(ylim)

			for tick in axes[i,j].get_xticklabels():
				tick.set_rotation(45)

	return

def hist_1d(ax, samples, bins, bounds, label, show_xticklabels):

	ax.hist(samples, bins=bins, range=bounds)
	ax.set_yticklabels([])
	ax.set_xlim(nice_bounds(samples))

	if show_xticklabels:
		ax.set_xlabel(label)
		ax.get_xaxis().set_major_locator(ticker.MaxNLocator(nbins=5, prune='upper'))
		labelOffset(ax, "x")
	else:
		ax.set_xticklabels([])
	
	return


def nice_bounds(samplesx, factor=3):
	"""Generate sensible limits for distribution plots.
	
	Finds the mean+factor*std_dev and mean-factor*std_dev of a set of samples.
	
	Parameters:
	----------
	samplesx : {ndarray}
		Samples from a distribution.
	factor : {int}, optional
		Number of standard deviations to includde. (the default is 3, which usually gives nice looking plots without being too zoomed out)
	
	Returns
	-------
	tuple
		(lower limit, upper limit) of plot.
	"""
	
	sx = factor*np.std(samplesx)
	avgx = np.mean(samplesx)
	return (avgx-sx, avgx+sx)

def walkerTrace(fig, samples, labels=None, **kwargs):
	"""Generate a walker trace figure from MCMC samples.
	
	Given some input MCMC samples, generate a figure with ndim subplots, one for each model parameter, showing the traces of each walker through the parameter subspace.
	
	Parameters:
	----------
	fig : {figure}
		Empty Matplotlib figure in which to draw the walker trace subplots.
	samples : {ndarray}
		Output of MCMC sampler, must be of shape (nwalkers, nsamples, ndim).
	labels : {list}, optional
		List of length *ndim* containing variable names for each parameter. (the default is None, which means your parameters are unlabeled.)
	
	"""


	nwalkers, nsteps, ndim = np.shape(samples)

	if labels is None:
		labels = [None for _ in range(ndim)]

	if "color" not in kwargs:
		kwargs["color"] = 'k'
	if "alpha" not in kwargs:
		kwargs["alpha"] = 0.3


	axes = fig.subplots(ndim, 1)
	fig.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.98, wspace=0.05, hspace=0.05)

	for i in range(ndim):
		axes[i].plot(samples[:,:,i].T, **kwargs)

		if i < ndim - 1:
			axes[i].set_xticklabels([])

		axes[i].set_xlim(0, nsteps)
		if labels[i] is not None:
			axes[i].set_ylabel(labels[i])

		labelOffset(axes[i], "y")

	axes[ndim-1].set_xlabel("Step")

	return