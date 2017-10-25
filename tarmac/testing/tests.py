import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import cmocean

import pytest
import tarmac.tarmac as tm

def test_corner():
	data = np.load("extra_data.npy")

	fig = plt.figure(figsize=(10, 10))
	tm.cornerPlot(fig, data.reshape((-1, 4)), labels=["a", "b", "c", "d"], cmap=cmocean.cm.tempo_r)
	# plt.savefig("original.png", dpi=300)
	plt.show()
	return


def test_walkerTrace():
	data = np.load("extra_data.npy")

	fig = plt.figure(figsize=(10, 10))
	tm.walkerTrace(fig, data, labels=["a", "b", "c", "d"])
	# plt.savefig("unpacked.png", dpi=300)
	plt.show()
	return