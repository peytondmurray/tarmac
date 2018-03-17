import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest, cmocean
import tarmac.tarmac as tm

def test_corner():
	"""Test the corner plot function.
	
	Construct a corner plot using the example data.
	
	"""


	data = np.load("extra_data.npy")

	fig = plt.figure(figsize=(10, 10))
	# tm.cornerPlot(fig, data.reshape((-1, 4)), labels=["a", "b", "c", "d"], cmap=cmocean.cm.cmap_d['tempo_r'])
	tm.corner_plot(fig, data, labels=['a','b','c','d'], cmap=cmocean.cm.cmap_d['tempo_r'])
	plt.show()
	return


def test_walkerTrace():
	"""Test the walker trace function.
	
	Construct a walker trace plot using the example data.
	
	"""

	data = np.load("extra_data.npy")

	fig = plt.figure(figsize=(10, 10))
	tm.walkerTrace(fig, data, labels=["a", "b", "c", "d"])
	plt.show()
	return
