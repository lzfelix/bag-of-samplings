import glob
from os import path
from typing import List, Tuple
from pathlib import Path

import tqdm
import numpy as np


def load_signal(filepath: str, sampling_factor: int = 0) -> np.ndarray:
    """Loads features from a single exam, returning a tensor of features.
    
    # Arguments
        filepath: The fully-specified path to the handPD2 textfile to load
        sampling_factor: Sample the timeseries at every t ms.
    # Returns
        A tensor with shape [? = n_sampled_timesteps, 6 = n_channels]
    """

    all_features = list()
    sampling_index = 0
    if sampling_factor < 1:
        sampling_factor = 1
    
    for line in open(filepath, 'r'):
        # Comment lines
        if line[0] == '#':
            continue

        if sampling_index % sampling_factor == 0:            
            features = line.split()
            all_features.append(features)
            sampling_index = 0
        sampling_index += 1
    return np.asarray(all_features, np.float32)


def load_fold_signals(filepath: str,
                      fold_id: int,
                      is_healthy: bool,
                      sampling_factor: int,
                      verbose: bool) -> List[np.ndarray]:

    fold = ['train', 'valid', 'test'][fold_id]
    kind = 'healthy' if is_healthy else 'patients'
    fold_filepath = Path(filepath) / kind / fold / '*.txt'

    all_exams_features = list()
    files = glob.glob(str(fold_filepath))
    files_iter = tqdm.tqdm(files) if verbose else files
    for file in files_iter:
        all_exams_features.append(load_signal(file, sampling_factor))
    return all_exams_features


def load_handp_v2(filepath: str,
                  fold_id: int,
                  sampling_factor: int,
                  verbose: bool = True) -> Tuple[List[np.ndarray], np.ndarray]:
    """Loads healthy *and* patient signals from train or test folds.

    # Arguments
        filepath: Path to the root folder of the dataset, ie: spirals_75_25.
        fold_id: 0 = train, 1 = dev, 2 = test
        sampling_factor:
    # Returns
        `X` and `y`, the first is a list of tensors with shape
        `[? = n_timsteps, 6 = n_channels]` and the second a tensor with shape `n_samples`.
        Note: both are unshuffled.
    """
    assert 0 <= fold_id <= 2, 'Fold number should be between 0 (train) and 2 (test)'

    healthy_signals = load_fold_signals(filepath, fold_id, True, sampling_factor, verbose)
    label_healthy = np.zeros(len(healthy_signals), dtype=np.int32)

    patient_signals = load_fold_signals(filepath, fold_id, False, sampling_factor, verbose)
    labels_patients = np.ones(len(patient_signals), dtype=np.int32)

    x = healthy_signals + patient_signals
    y = np.concatenate([label_healthy, labels_patients], axis=0)
    return x, y


def load_all_exams(root_folder: str, exam: str, sampling_factor: int) -> List[np.ndarray]:
    """Reads all exams from a folder
    
    # Arguments
        root_folder: Where all the samples are stored
        exam: The exam acronym (sp: spiral, circ: circle, mea: meander)
        sampling_factor: sample a timeseries at every [sampling_factor] miliseconds
    # Returns
        A list L of tensors where L[i] is the sampled signal from the i-th sample.
        L[i].shape == (? timesteps, 6 channels)
    """
    
    # for instance: data/healthy/sigSp*.txt
    all_exams_path = path.join(root_folder, exam + '*.txt')
    all_exams = glob.glob(all_exams_path)

    all_exams_features = list()
    for i in tqdm.trange(len(all_exams)):
        exam_file = all_exams[i]
        all_exams_features.append(load_signal(exam_file, sampling_factor))
    return all_exams_features
