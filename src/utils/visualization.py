import matplotlib.pyplot as plt
import numpy as np


def plot_results(series, labels, errors, thr, title=""):

    plt.figure(figsize=(12, 4))
    plt.plot(series)

    plt.scatter(
        np.where(labels == 1)[0],
        series[labels == 1],
        c="red",
        s=10
    )

    plt.title(title + " Signal")
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(errors)
    plt.axhline(thr, color="red", linestyle="--")

    plt.title(title + " Error")
    plt.grid()
    plt.show()