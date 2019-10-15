import sys
from typing import Tuple, List
from collections import defaultdict

import numpy as np
from keras.preprocessing import sequence

Pair = Tuple[float, float]
PairList = List[Pair]


def clip_values(samples: List[np.ndarray],
                cutoffs: List[Pair]) -> List[np.ndarray]:
    """Clips all values to their boundaries according to cutoffs.

    # Arguments
        samples: A list of tensor with shape [? = n_timesteps, n_channels]
        cutoffs: A list of (lower, upper) bound values for each channel.
    # Returns
        A new list (operations are not performed inplace) such that
        `cutoff[channel] < samples[i, :, channel] < cutoff[channel][1]`.
    """
    clipped_samples = list()
    n_channels = samples[0].shape[-1]
    assert len(cutoffs) == n_channels

    for sample in samples:
        new_sample = sample.copy()

        for channel in range(n_channels):
            lower, upper = cutoffs[channel]

            new_sample[new_sample[:, channel] < lower, channel] = lower
            new_sample[new_sample[:, channel] > upper, channel] = upper
        clipped_samples.append(new_sample)
    return clipped_samples


def mean_std(X: List[np.ndarray], channel: int) -> Pair:
    """Given a list of timeseries with shape [timesteps, n_channel], computes
    avg and stddev for a channel.
    
    # Arguments
        X: List of timeseries tensors.
        channel: The desired channel to have the statistics computed
    # Returns
        (avg, stddev)
    """
    accumulator = 0
    lengths_sum = 0
    for sample in X:
        accumulator += np.sum(sample[:, channel])
        lengths_sum += sample.shape[0]

    avg = accumulator / lengths_sum

    # stddev = sqrt(sum(samples[:, :, channel] - avg[channel]) / sum(lengths))
    accumulator = 0
    for sample in X:
        diff = sample[:, channel] - avg
        accumulator += np.sum(np.power(diff, 2))
    stddev = np.sqrt(accumulator / lengths_sum)
    
    return avg, stddev


def min_max(X: np.ndarray, channel: int) -> Pair:
    """Given a list of timeseries with shape [timesteps, n_channel], computes
    min and (max-min) for a channel.

    # Arguments
        X: List of timeseries tensors.
        channel: The desired channel to have the statistics computed
    # Returns
        (min, max - min)
    """
    channel_max = 0
    channel_min = np.inf
    for sample in X:
        current_max = np.max(sample[:, channel])
        current_min = np.min(sample[:, channel])

        channel_max = max(current_max, channel_max)
        channel_min = min(current_min, channel_min)

    return channel_min, (channel_max - channel_min)


def compute_series_stats(samples: List[np.ndarray],
                         fun) -> List[Tuple[float, float]]:
    """Computes some statistics over samples using a provided function.

    # Arguments
        samples: A list of timeseries with shape [? = n_timsteps, n_channels].
        fun: A callable that takes all samples and computes some metrics over
            one of its channels. This function is sequentially over all channels
            of the samples.
    # Returns
        A list with the metrics computed for each of the n_channels in all samples
            using fun.
    """
    channel_stats = list()

    n_channels = samples[0].shape[-1]
    for channel in range(n_channels):
        a, b = fun(samples, channel)
        channel_stats.append((a, b))
    return channel_stats


def normalize_multiseries(samples: List[np.ndarray],
                          statistics: List[Tuple[float, float]]) -> List[np.ndarray]:
    """Normalizes all channels in multiseries with `(x[i, t, c] - a[c]) / b[c])`.
    # Arguments
        multiseries: Tensor with shape [n_samples, n_timesteps, n_channels].
        statistics: A list of tuples (a, b) so that each normalized multiseries is
            computed as `(x[i, t, c] - a[c]) / b[c])`.
    # Returns
        normalized_sampels: samples after being normalized with statistcs.
    """
    # Copy because these operations happen inplace and the original tensor is needed.
    normalized_samples = list()
    n_channels = samples[0].shape[-1]

    for sample in samples:
        new_sample = np.zeros_like(sample)
        for channel in range(n_channels):
            a, b = statistics[channel]
            new_sample[:, channel] = (sample[:, channel] - a) / b

        normalized_samples.append(new_sample)
    return normalized_samples


def sampling(sequence: np.ndarray, sampling_factor: int) -> np.ndarray:
    """Samples a timeseries at every sample_factor ms.
    
    # Arguments
        sequence: Tensor with shape `[n_timesteps, n_channels]`.
        sampling_factor: Samples sequence at every given ms. If sampling is negative,
            the sampling is performed from the end to the begining of the sequence.
    # Return
        A tensor with shape [lower(n_timesteps // sampling_factor), n_channels]
    """
    if sampling_factor < 0:
        sampling_factor *= -1
        sequence = np.flip(sequence, axis=0)

    sampled = list()
    for index, timestep in enumerate(sequence):
        if index % sampling_factor == 0:
            sampled.append(timestep)
    return np.asarray(sampled)


def equal_sampling(sequence: np.ndarray, resulting_seq_len: int) -> np.ndarray:
    # Grants that the resulting sequence is not longer than the original sequence
    series_len = sequence.shape[0]
    resulting_seq_len = min(resulting_seq_len, series_len)

    sampling_t = np.linspace(0, series_len - 1, resulting_seq_len, dtype=int)
    return sequence[sampling_t]


def generate_sampled_signals(samples: List[np.ndarray],
                             sampling_factor: int,
                             maxlen: int) -> Tuple[np.ndarray, List[int]]:
    """Samples all timeseries in signals_ at every sampling_factor ms.
    
    # Arguments
        signals_: A list of Tensors with shape [?, n_channels].
        sampling_factor: Samples the sequences at every given ms.
            If this value is negative, the sampling is performed from
            the end to the begining of the sequence.
        maxlen: Pads the sampled sequences to have maxlen timesteps.
        strategy: ...

    # Returns
        sampled: A tensor with shape [n_samples, maxlen, n_channels].
        sampled_lengths: The length of each signal before padding
    """
    if sampling_factor != 0:
        sampled = [sampling(signal, sampling_factor) for signal in samples]
    else:
        sampled = [equal_sampling(signal, maxlen) for signal in samples]

    sampled_lengths = [min(len(signal), maxlen) for signal in sampled]

    sampled = sequence.pad_sequences(sampled,
                                     maxlen=maxlen,
                                     dtype='float32',
                                     padding='post',
                                     truncating='post')
    return sampled, sampled_lengths


def multires_sampling(x_trn: List[np.ndarray],
                      x_val: List[np.ndarray],
                      x_tst: List[np.ndarray],
                      resolutions: List[int],
                      maxlens: List[int]) -> Tuple[defaultdict, defaultdict]:
    """Samples train, validation and test signals using different resolutions.
    # Arguments
        x_trn: A list of timeseries samples without any previous sampling.
        x_val: A list of timeseries samples without any previous sampling.
        x_tst: A list of timeseries samples without any previous sampling.
        resolutions: A list of resolutions to perform samplings.
        mexlens: For each resolution, the default length for its sampled versions.
    # Returns
        `A` is a dict with the following layout for train, validation and test sets:
        ```python
            d['train'][f'res_{resolutions[i]}'] ->
                np.ndarray with shape [n_samples, maxlen, n_channels]
        ```

        `B` contains the length of each signal before padding.
    """
    assert len(resolutions) == len(maxlens)
    if resolutions[0] == 0 and len(resolutions) == 1:
        print('[WARN] Using equally-spaced sampling')
        sys.stdout.flush()
    else:
        print('xxx')

    bag_of_resolutions = defaultdict(dict)
    bag_of_lengths = defaultdict(dict)
    for resolution, maxlen in zip(resolutions, maxlens):
        trn_samples, trn_lens = generate_sampled_signals(x_trn, resolution, maxlen)
        val_samples, val_lens = generate_sampled_signals(x_val, resolution, maxlen)
        tst_samples, tst_lens = generate_sampled_signals(x_tst, resolution, maxlen)

        res_name = f'res_{resolution}'
        bag_of_resolutions['trn'][res_name] = trn_samples
        bag_of_resolutions['val'][res_name] = val_samples
        bag_of_resolutions['tst'][res_name] = tst_samples

        bag_of_lengths['trn'][res_name] = trn_lens
        bag_of_lengths['val'][res_name] = val_lens
        bag_of_lengths['tst'][res_name] = tst_lens

    return bag_of_resolutions, bag_of_lengths
