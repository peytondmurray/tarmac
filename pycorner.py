import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import cmocean
import matplotlib.ticker as ticker

def labelOffset(ax, axis="y"):

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

	def updateLabel(lim):
		offset = fmt.get_offset()
		if offset != '':
			offset = "({})".format(offset)
		labelfunc("{} {}".format(label, offset))
		return

	ax.callbacks.connect("ylim_changed", updateLabel)
	ax.figure.canvas.draw()
	updateLabel(None)
	return

def cornerPlot(fig, samples, bins=100, ranges=None, labels=None, cmap=None, plotType="hist"):
	"""
	samples: 2D array of shape (nsamples, ndim)
	"""


	#Make sure the arguments are all copacetic
	assert len(np.shape(samples)) == 2, "samples list must be of shape (nsamples, ndim), but is of shape {}".format(np.shape(samples))
	nsamples, ndim = np.shape(samples)
	assert nsamples > ndim, "Number of samples is greater than number of dimensions."


	if isinstance(bins, int):
		bins = np.array([bins for _ in range(ndim)])
	else:
		assert len(np.shape(bins)) == 1, "Bins should be a 1D array or an integer."
		assert np.shape(bins)[0] == ndim, "Dimension mismatch between bins and number of columns in samples."

	if ranges is None:
		ranges = [makeNiceLimits(samples[:,i]) for i in range(ndim)]
	else:
		assert len(ranges) == ndim, "Dimension mismatch between ranges and number of columns in samples."
		for i in range(len(ranges)):
			if range[i] is None:
				range[i] = [np.nanmin(samples[:,i]), np.nanmax(samples[:,i])]

	#Set the default colormap to viridis
	if cmap is None:
		cmap = "viridis"

	#Divide the figure into a bunch of subplots, and Remove whitespace between plots
	axes = fig.subplots(ndim, ndim)
	fig.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.98, wspace=0.05, hspace=0.05)

	for i in range(ndim):

		#Plot the 1D histograms along the diagonal
		ax = axes[i, i]
		ax.hist(samples[:,i], bins=bins[i], range=ranges[i])
		ax.set_yticklabels([])
		ax.set_xlim(makeNiceLimits(samples[:,i]))
		if i < ndim - 1:
			ax.set_xticklabels([])
		elif i == ndim - 1:
			ax.set_xlabel(labels[i])
			ax.get_xaxis().set_major_locator(ticker.MaxNLocator(nbins=5, prune='upper'))


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
				xlim, ylim = makeNiceLimits(samples[:,j]), makeNiceLimits(samples[:,i])
				axes[i,j].set_xlim(xlim)
				axes[i,j].set_ylim(ylim)

			for tick in axes[i,j].get_xticklabels():
				tick.set_rotation(45)


	return

def makeNiceLimits(samplesx, factor=3):
	sx = factor*np.std(samplesx)
	avgx = np.mean(samplesx)
	return [avgx-sx, avgx+sx]

def walkerTrace(fig, samples, labels=None, **kwargs):
	#Make sure the arguments are all copacetic
	# assert len(np.shape(samples)) == 2, "samples list must be of shape (nsamples, ndim), but is of shape {}".format(np.shape(samples))
	nwalkers, nsteps, ndim = np.shape(samples)
	# assert nsamples > ndim, "Number of samples is greater than number of dimensions."

	axes = fig.subplots(ndim, 1)
	fig.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.98, wspace=0.05, hspace=0.05)

	for i in range(ndim):
		axes[i].plot(samples[:,:,i], **kwargs)

		if i < ndim - 1:
			axes[i].set_xticklabels([])


		axes[i].set_xlim(0, len(samples[:,i]))
		labelOffset(axes[i], "y")
		for tick in axes[i].get_xticklabels():
			tick.set_rotation(45)

	return

if __name__ == "__main__":
	fig = plt.figure(figsize=(10,10))

	with open("testing/test_data.csv", 'r') as f:
		rawdata = f.readlines()

	data = []
	for line in rawdata:
		data.append([float(pt) for pt in line.split(',')])

	data = np.array(data)

	cornerPlot(fig, data, labels=["a", "b", "c", "d"], plotType="hist", cmap=cmocean.cm.tempo_r)
	# walkerTrace(fig, data.reshape((-1,20,4)), labels=["a", "b", "c", "d"], linestyle='-', color='k', alpha=0.3)
	plt.show()