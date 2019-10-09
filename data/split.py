import os
import glob
import shutil
import argparse
from typing import List, Tuple
from pathlib import Path

import tqdm
import numpy as np

# Ensuring reproducible, deterministic dataset partitions
np.random.seed(32)


def shuffle_slice_dir(folder: str,
                      train_percentage: float,
                      dev_percentage: float) -> Tuple[List[str], List[str], List[str]]:
    """Shuffles files and splits them into train and test.
    # Arguments
        folder: Folder to the path of samples, eg: ./patients/exam*.txt.
        train_percentage:
        dev_percentage:
    # Return
        Three lists of filenames, with train, dev and test filepaths.
    """
    all_files = glob.glob(folder)
    np.random.shuffle(all_files)

    n = len(all_files)
    train_pivot = int(n * train_percentage)
    dev_pivot = train_pivot + int(n * dev_percentage)

    return all_files[:train_pivot], all_files[train_pivot:dev_pivot], all_files[dev_pivot:]


def copy_files(parent_folder: str,
               kind: str,
               train_files: List[str],
               dev_files: List[str],
               test_files: List[str]) -> None:
    """Given train and test files, allocates them into folders.
    # Arguments
        parent_folder: The dataset root folder to be created.
        kind: Either "patient" or "healthy".
        train_files: A list of files to be copied into
            `parent_folder/kind/train`.
        dev_files: A list of files to be copied into
            `parent_folder/kind/dev`.
        test_files: A list of files to be copied into
            `parent_folder/kind/test`.
    # Returns
        None
    """
    # Create the kind of split folder, ie spiral_50_50
    parent_folder = Path(parent_folder)
    parent_folder.mkdir(exist_ok=True)
    
    # Create the type of sample folder (patient, control)
    parent_folder = parent_folder / kind
    parent_folder.mkdir(exist_ok=True)
    
    def move_child(folder_name, files):
        child = (parent_folder / folder_name)
        child.mkdir(exist_ok=True)
        for file in tqdm.tqdm(files):
            shutil.copy(file, child)
    
    # Move the samples to the type of sample folder
    move_child('train', train_files)
    move_child('valid', dev_files)
    move_child('test', test_files)


def split_samples(kind: str,
                  exam: str,
                  perc_trn: float,
                  perc_dev: float) -> None:
    """Given a list of filepaths, splits them into train/dev/test.
    # Arguments
        kind: Either 'patient' or 'healthy'.
        exam: Either meander (Mea), Spiral (Sp) or Circle (circ).
        perc_trn: *integer* representing how many samples are reserved
            for training.
    # Returns
        None
    """
    train_pd, dev_pd, test_pd = shuffle_slice_dir(f'./{kind}/sig{exam}*.txt',
                                                  perc_trn, perc_dev)
    total_samples = len(train_pd) + len(dev_pd) + len(test_pd)
    print(f'[{kind} / {exam}]')
    print(f'Trn:   {len(train_pd)} samples')
    print(f'Dev:   {len(dev_pd)}   samples')
    print(f'Tst:   {len(test_pd)}  samples')
    print(f'Total: {total_samples} samples')
    
    amount_train = int((perc_trn + perc_dev) * 100)

    copy_files(f'./_{exam.lower()}_{amount_train}_{100 - amount_train}',
               kind,
               train_pd,
               dev_pd,
               test_pd)


def get_arguments() -> Tuple[str, float, float]:
    parser = argparse.ArgumentParser(usage='Utility script to split the HandPD_v2 dataset.')
    parser.add_argument('exam', help='Either Mea (meanter), Circ (circle), Sp (spiral). ' +
                                     'Case sensitive.')
    parser.add_argument('trn_frac', help='Fraction of samples used for training. 0 < x < 1.', type=float)
    parser.add_argument('val_frac', help='Fraction of samples used for validation. 0 < x < 1.', type=float)

    args = parser.parse_args()
                        
    valid_exams = ['Sp', 'Mea', 'Circ']
    if args.exam not in valid_exams:
        raise ValueError('Valid exams are {}'.format(', '.join(valid_exams)))

    trn_frac = args.trn_frac
    val_frac = args.val_frac
    
    if (not 0 < trn_frac < 1) or (not 0 < val_frac < 1) or (trn_frac + val_frac >= 1):
        raise RuntimeError('Percentages should be between 0 and 1. Their sum must be smaller than 1')

    return args.exam, trn_frac, val_frac


if __name__ == '__main__':
    exam, trn_frac, dev_frac = get_arguments()

    split_samples('patients', exam, trn_frac, dev_frac)
    split_samples('healthy', exam, trn_frac, dev_frac)
