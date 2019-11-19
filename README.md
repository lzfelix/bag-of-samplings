# Bag of Samplings
_Official implementation of the paper ["Bag of Samplings for Parkinson's Disease Diagnosis based on Recurrent Neural Networks"](https://www.sciencedirect.com/science/article/pii/S0010482519303476)_


## Reproducing the results

1. Install the requirements with `pip install -r requirements.txt` (**requires Python 3.6**)
2. Run `./data/get_handpd2.sh` to download the data from the [Recogna Laboratory](http://www.recogna.tech) servers
3. The training protocols are based in the work by Afonso et al. `[1]`, in which two training regimes are considered: a) 50% of the data is used for training and 50% for testing b) 75% of the data is used for training and 25% for testing. In our case we need to validate the hyperparameters, hence we employ the data partitions in the table below. 

| Training regime | Train | Validation | Test |
| --------------- | ----- | ---------- | ---- |
| a               | 50%   | 0%         | 50%  |
| b               | 75%   | 0%         | 25%  |
| a (ours)        | 40%   | 10%        | 50%  |
| b (ours)        | 65%   | 10%        | 25%  |

These splits can be generated with the script `split.py` in `data/`. Since the random seed is fixed, the output of the script should always be the same. Nevertheless, the splits used in our experiments are detailed in the file `data_splits.py`.

4. You can reproduce the paper results using the script `models/train_models.py`. Running it with the `-h` option shows how to set the hyperparameters.
5. If you wish to run the sampling interval powerset selection, use `models/poewrset_selection.py` the output of this script is persisted in the disk.

## Learning curve

To reproduce the experiment depicted in Figure 6 in our paper, run `experiments/models/learning_curve.py` and to generate the plots, use `experiments/models/plot_learning_curves.py`.

## Citation

If you use our work in your research, please cite the following paper:

> Ribeiro, L.C., Afonso, L.C., Papa, J.P.. [Bag of samplings for computer-assisted parkinson’s disease diagnosis based on recurrent neural networks](https://www.sciencedirect.com/science/article/pii/S0010482519303476).

```
@article{ribeiro2019bos,
  title = "Bag of Samplings for computer-assisted Parkinson's disease diagnosis based on Recurrent Neural Networks",
  journal = "Computers in Biology and Medicine",
  volume = "115",
  pages = "103477",
  year = "2019",
  issn = "0010-4825",
  doi = "https://doi.org/10.1016/j.compbiomed.2019.103477",
  url = "http://www.sciencedirect.com/science/article/pii/S0010482519303476",
  author = "Luiz C.F. Ribeiro and Luis C.S. Afonso and João P. Papa",
}
```


## References

```
[1] L. C. Afonso, G. H. Rosa, C. R. Pereira, S. A. Weber, C. Hook, V. H. C. Albuquerque, and J. P. Papa, “A recurrence plot-based approach for parkinson’s disease identification,” Future Generation Computer Systems, vol. 94, pp. 282 – 292, 2019.
```
