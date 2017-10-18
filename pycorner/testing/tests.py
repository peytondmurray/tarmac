import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import cmocean
from tqdm import tqdm

import pytest
import pycorner.pycorner as pc

def test_corner():
	data = np.load("extra_data.npy")

	fig = plt.figure(figsize=(10, 10))
	pc.cornerPlot(fig, data.reshape((-1, 4)), labels=["a", "b", "c", "d"], plotType="hist", cmap=cmocean.cm.tempo_r)
	# plt.savefig("original.png", dpi=300)
	plt.show()
	return


def test_walkerTrace():
	data = np.load("extra_data.npy")

	fig = plt.figure(figsize=(10, 10))
	pc.walkerTrace(fig, data, linestyle='-', color='k', alpha=0.3)
	# plt.savefig("unpacked.png", dpi=300)
	plt.show()
	return