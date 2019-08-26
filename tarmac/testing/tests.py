import tarmac.tarmac as tm
import numpy as np
import pytest
import matplotlib.pyplot as plt


def test_corner_plot():
    """Test the corner plot function.
    
    Construct a corner plot using the example data.
    
    """

    data = np.load("extra_data.npy")

    fig = plt.figure(figsize=(10, 10))

    tm.corner_plot(fig,
                   data,
                   labels=['a','b','c','d'],
                   cmap='viridis'
                   )
    plt.show()
    return


def test_walker_trace():
    """Test the walker trace function.
    
    Construct a walker trace plot using the example data.
    
    """

    data = np.load("extra_data.npy")

    fig = plt.figure(figsize=(10, 10))
    tm.walker_trace(fig, data, labels=["a", "b", "c", "d"])
    plt.show()
    return
