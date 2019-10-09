import numpy as np

def average_channel_from_all_samples(X, X_lenghts, channel):
    """Given a timeseries as a tensor with shape (samples, time, channel), computes avg and stddev
    for each channel without taking into account the padding values.
    
    # Arguments
        X: The timeseries tensor
        X_lenghts: A list with the lenght of each timeseries before padding
        channel: The desired channel to have the statistics computed

    # Return
        (avg, stddev)
    """
    accumulator = 0
    lengths_sum = 0
    for sample, length in zip(X, X_lenghts):
        accumulator += np.sum(sample[:length, channel])
        lengths_sum += length

    avg = accumulator / lengths_sum
    
    accumulator = 0
    for sample, lenght in zip(X, X_lenghts):
        diff = sample[:length, channel] - avg
        
        accumulator += np.sum(np.power(diff, 2))
    stddev = np.sqrt(accumulator / lengths_sum)
    
    return avg, stddev


def compute_channels_mean_std(multiseries, samples_lengths):
    n_channels = multiseries.shape[-1]
    mus = np.zeros(n_channels)
    sigmas = np.zeros(n_channels)
    
    for channel in range(n_channels):
        mu, sigma = average_channel_from_all_samples(multiseries, samples_lengths, channel)
        mus[channel] = mu
        sigmas[channel] = sigma
    return mus, sigmas


def normalize_partial_multiseries(multiseries, samples_lenghts, mus, sigmas):
    normalized = multiseries.copy()
    
    n_channels = multiseries.shape[-1]
    for channel in range(n_channels):
        normalized[:,:, channel] = (multiseries[:, :, channel] - mus[channel]) / sigmas[channel]

    # First normalize, then ignore normalization on padded timesteps
    mask = np.asarray(multiseries > 0, dtype=np.float)
    return (normalized * mask)


def normalize_multiseries(multiseries, samples_lenghts):
    normalized = multiseries.copy()
    
    n_channels = multiseries.shape[-1]
    for channel in range(n_channels):
        mu, sigma = average_channel_from_all_samples(multiseries, samples_lenghts, channel)
        normalized[:,:, channel] = (multiseries[:, :, channel] - mu) / sigma

    # First normalize, then ignore normalization on padded timesteps
    mask = np.asarray(multiseries > 0, dtype=np.float)
    return (normalized * mask)