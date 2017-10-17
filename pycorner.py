import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np


def pycorner(samples, bins=40, ranges=None, labels=None, fig=None, cmap=None, plotType="hist"):
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

	# Create a new figure if one wasn't provided.
	if fig is None:
		fig, axes = plt.subplots(ndim, ndim, figsize=(ndim, ndim))
	else:
		try:
			axes = np.array(fig.axes).reshape((ndim, ndim))
		except:
			raise ValueError("Provided figure has {} axes, but data has dimensions K={}".format(len(fig.axes), ndim))

	#Set the default colormap to viridis
	if cmap is None:
		cmap = "viridis"

	#Remove whitespace between plots
	fig.subplots_adjust(left=0.1, bottom=0.1, right=0.98, top=0.98, wspace=0.05, hspace=0.05)

	for i in range(ndim):

		#Get the axes
		ax = axes[i, i]
		ax.hist(samples[:,i], bins=bins[i], range=ranges[i])
		ax.set_yticklabels([])
		ax.set_xlim(makeNiceLimits(samples[:,i]))
		if i < ndim - 1:
			ax.set_xticklabels([])
		elif i == ndim - 1:
			ax.set_xlabel(labels[i])

		for j in range(ndim):

			if j > i:
				axes[i,j].axis('off')


			if j < i:
				if plotType in ["hist", "histogram"]:
					axes[i,j].hist2d(samples[:,j], samples[:,i], bins=[bins[j], bins[i]], range=[ranges[j], ranges[i]], cmap=cmap)
				elif plotType in ["hex", "hexbin"]:
					axes[i,j].hexbin(samples[:,j], samples[:,i], gridsize=[bins[j], bins[i]], extent=[*ranges[j], *ranges[i]], cmap=cmap)
				if j == 0:
					axes[i,j].set_ylabel(labels[i])
				else:
					axes[i,j].set_yticklabels([])
				if i == ndim - 1:
					axes[i,j].set_xlabel(labels[j])
				else:
					axes[i,j].set_xticklabels([])

				for tick in axes[i,j].get_xticklabels():
					tick.set_rotation(45)

				xlim, ylim = makeNiceLimits(samples[:,j]), makeNiceLimits(samples[:,i])

				axes[i,j].set_xlim(xlim)
				axes[i,j].set_ylim(ylim)
	return

def makeNiceLimits(samplesx, factor=2):
	sx = factor*np.std(samplesx)
	avgx = np.mean(samplesx)


	return [avgx-sx, avgx+sx]

if __name__ == "__main__":
	with open("test_data.csv", 'r') as f:
		rawdata = f.readlines()

	data = []
	for line in rawdata:
		data.append([float(pt) for pt in line.split(',')])

	data = np.array(data)

	pycorner(data, labels=["a", "b", "c", "d"], plotType="hex")
	plt.show()