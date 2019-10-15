from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def plot_sample(series: np.ndarray, create_fig: bool=True) -> None:
    """Plots all channels of a single timeseries."""
    if create_fig:
        plt.figure(figsize=(15, 5))
    plt.grid(alpha=0.4)

    n_channels = series.shape[-1]
    for i in range(n_channels):
        plt.plot(series[:, i], label=f'Channel {i}')
    plt.legend()


def ratios(split_name: str, labels: np.ndarray, dataset_size: int) -> None:
    """Prints the frequency of each class in each split.
    
    # Arguments
        split_name: Name to be displayed.
        labels: Labels for the current class (should be 0 or 1).
        dataset_size: Total size of the *dataset.
    """
    c = Counter(labels)
    z = sum(c.values())
    print(f'[{split_name}]')
    print('label\tfreq\t%')
    for label, freq in sorted(c.items()):
        print('{}\t{}\t{:2.2}'.format(label, freq, freq/z))
    print('%total\t\t{:2.3}'.format(z/dataset_size * 100))
    print()
