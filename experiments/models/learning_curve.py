import argparse
import sys
import train_models


def get_exec_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage='Trains models with fractions of the dataset')
    parser.add_argument('ds', help='Dataset. Either "sp" or "mea"', type=str)
    parser.add_argument('--start', help='Starts with this amount of data', default=5, type=int)
    parser.add_argument('--end', help='Uses at most this amount of data', default=100, type=int)
    parser.add_argument('--tick', help='Increases the progress by this amount of data', default=5, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    exec_args = get_exec_args()

    if exec_args.ds == 'sp':
        resolutions = [25, 50]
        lengths = [500, 500]
    elif exec_args.ds == 'mea':
        resolutions = [5, 25, 50, 100]
        lengths = [500, 500, 500, 500]
    else:
        raise ValueError('Valid exams are "sp" and "mea".')

    print(exec_args)
    print('Res:  ', resolutions)
    print('Lens: ', lengths)

    for train_frac in range(exec_args.start, exec_args.end + 1, exec_args.tick):
        print(f'Training with {train_frac}% of data.')
        sys.stdout.flush()
        
        train_frac /= 100
        train_models.train_models(exam=exec_args.ds,
                                  n_models=5,
                                  dropout=0.5,
                                  resolutions=resolutions,
                                  lengths=lengths,
                                  n_recurrent_layers=2,
                                  is_bidirectional=True,
                                  hidden_sz=64,
                                  n_epochs=30,
                                  ratio='75_25',
                                  trn_fraction=train_frac)
