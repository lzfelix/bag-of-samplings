import pickle
from typing import List, Tuple, Dict
from pathlib import Path

from sklearn import metrics
import numpy as np

prediction_files = ['amodel_2019-05-26 15:57:26.767185', 'amodel_2019-05-26 16:03:40.344077', 'amodel_2019-05-26 16:10:04.916788', 'amodel_2019-05-26 16:16:44.567356', 'amodel_2019-05-26 16:23:40.361160', 'amodel_2019-05-26 16:30:51.260939', 'amodel_2019-05-26 16:38:19.128682', 'amodel_2019-05-26 16:46:03.301254', 'amodel_2019-05-26 16:54:05.715967', 'amodel_2019-05-26 17:02:25.360698', 'amodel_2019-05-26 17:11:05.878199', 'amodel_2019-05-26 17:20:02.700648', 'amodel_2019-05-26 17:29:15.640363', 'amodel_2019-05-26 17:38:50.158162', 'amodel_2019-05-26 17:48:42.755205', 'amodel_2019-05-26 17:58:51.455916', 'amodel_2019-05-26 18:09:25.278040', 'amodel_2019-05-26 18:20:14.043582', 'amodel_2019-05-26 18:31:31.526811', 'amodel_2019-05-26 18:42:58.125302']

prediction_files = ['./preds/' + filename for filename in prediction_files]


def compute_metrics(data: Dict[str, np.ndarray],
                    partition: str) -> List[float]:
    predicted = (data[f'y_hat_{partition}'] >= 0.5).astype(int).flatten()
    ground = data[f'y_{partition}']
    tn, fp, fn, tp = metrics.confusion_matrix(ground, predicted).ravel()
    
    acc = (tp + tn) / (tp + tn + fp + fn)
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1  = (2 * pre * rec) / (pre + rec)
    acc_hc = tn / (tn + fn)
    acc_pd = tp / (tp + fp)
    return [acc, pre, rec, f1, acc_hc, acc_pd]


def store_metrics(buckets: dict, spreadables: List[float]) -> dict:
    for buck, metric in zip(buckets.keys(), spreadables):
        buckets[buck].append(metric)
    return buckets


def basic_stats(rol: List[float]) -> Tuple[float, float, float, float]:
    return np.average(rol), np.std(rol), np.max(rol), np.min(rol)


def print_report(prediction_files: List[str]) -> None:
    for split in ['val', 'tst']:
        all_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'acc_hc': [],
            'acc_pd': []
        }
    
        for file in prediction_files:
            data = pickle.load(Path(file).open('rb'))
            instance_metrics = compute_metrics(data, split)
            store_metrics(all_metrics, instance_metrics)

        print(f'[Split: {split}]')
        for name, observations in all_metrics.items():
            print('\t{:12}  {:4.4} ± {:4.4} {:4.4} {:4.4}'.format(name, *basic_stats(observations)))


if __name__ == '__main__':
    print_report(prediction_files)
