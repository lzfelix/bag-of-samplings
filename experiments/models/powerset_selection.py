# Finds the best combination of sampling intervals to use based
# on powerset (i.e. bruteforce) optimization.
# The obtained results are displayed on screen and persisted into
# disk as ./powerset_mea.pik

import pickle
import pprint
import itertools
from typing import List
from pathlib import Path

import train_models

import numpy as np
from scipy import special
from opytimizer.spaces.search import SearchSpace
from opytimizer.optimizers import bha
from opytimizer.core.function import Function
from opytimizer import Opytimizer


FIXED_NL = 2              # Amount of hidden layers
FIXED_BD = True           # Replace GRU by biGRU
FIXED_SZ = 64             # RNN layer size
FIXED_DROPOUT = 0.5       # dropout
FIXED_N_EPOCHS = 30       # Amount of epochs to train the model
N_MODELS_ESTIMATION = 1   # amount of models to estimate fitness
AVAILABLE_RES = [5, 25, 50, 100, 150]


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


def fit_power_model(resolutions: List[int]) -> float:
    val_acc = train_models.train_models(
        exam='mea',
        n_models=N_MODELS_ESTIMATION,
        dropout=FIXED_DROPOUT,
        resolutions=resolutions,
        lenghts=[500] * len(resolutions),
        n_recurrent_layers=FIXED_NL,
        is_bidirectional=FIXED_BD,
        hidden_sz=FIXED_SZ,
        n_epochs=FIXED_N_EPOCHS,
        verbose=False
    )
    return val_acc


if __name__ == '__main__':
    powerset = list(powerset(AVAILABLE_RES))
    powerset = powerset[1:]  # Dropping the emtpy set of resolutions
    
    fx = dict()
    n_res = len(powerset)
    for i, resolutions in enumerate(powerset):
        print('>>> Current resolution/s {}/{} [{}])'.format(
              i+1,
              n_res+1,
              ','.join(map(str, resolutions))))
        fx[resolutions] = fit_power_model(resolutions)

    pprint.pprint(fx, indent=4)
    pickle.dump(fx, Path('./powerset_mea.pik').open('wb'))

