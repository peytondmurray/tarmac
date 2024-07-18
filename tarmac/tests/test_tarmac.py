import matplotlib.pyplot as plt
import numpy as np
import pytest

import tarmac


@pytest.fixture(scope="module")
def example_data():
    """Load example data for plotting."""
    return np.load("extra_data.npy")


def test_corner_plot(example_data):
    """Test that a corner plot can be constructed."""
    fig = plt.figure(figsize=(10, 10))
    tarmac.corner_plot(
        fig,
        example_data,
        labels=["a", "b", "c", "d"],
        cmap="viridis",
    )


def test_walker_trace(example_data):
    """Test that a walker trace can be constructed."""
    fig = plt.figure(figsize=(10, 10))
    tarmac.walker_trace(
        fig,
        example_data,
        labels=["a", "b", "c", "d"],
    )
