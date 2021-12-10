# MOCCA: Multi-Layer One-Class ClassificAtion for Anomaly Detection


<p align="center">
<img src="https://github.com/fvmassoli/mocca-anomaly-detection/blob/main/images/mocca.png"  alt="MOCCA" width="700" height="450">
</p>



This repository contains the code relative to the paper "[MOCCA: Multi-Layer One-Class ClassificAtion for Anomaly Detection](https://ieeexplore.ieee.org/document/9640579)" by Fabio Valerio Massoli (ISTI - CNR), Fabrizio Falchi (ISTI - CNR), Alperen Kantarci (ITU), Şeymanur Akti (ITU), Hazim Kemal Ekenel (ITU), Giuseppe Amato (ISTI - CNR).

It reports a new technique to detect anomalies based on a layer-wise paradigm to exploit the features maps generated at different depths of a Deep Learning model.

The paper has been accepted for publication in the [IEEE Transactions on Neural Networks and Learning Systems, Special Issue on Deep Learning for Anomaly Detection](https://ieeexplore.ieee.org/document/9640579).

DOI: [10.1109/TNNLS.2021.3130074](https://doi.org/10.1109/TNNLS.2021.3130074).

**Please note:** 
We are researchers, not a software company, and have no personnel devoted to documenting and maintaing this research code. Therefore this code is offered "AS IS". Exact reproduction of the numbers in the paper depends on exact reproduction of many factors, including the version of all software dependencies and the choice of underlying hardware (GPU model, etc). Therefore you should expect to need to re-tune your hyperparameters slightly for your new setup.


## How to run the code

Before to run the code, make sure that your system has the proper packages installed. You can have a look at the [requirements.txt](https://github.com/fvmassoli/mocca-anomaly-detection/blob/main/requirements.txt) file.


Minimal usage (CIFAR10):

```
python main_cifar10.py -ptr -tr -tt -zl 128 -nc <normal class> -dp <path to CIFAR10 dataset>
```

Minimal usage (MVTec):

```
python main_mvtec.py -ptr -tr -tt -zl 128 -nc <normal class> -dp <path to MVTec dataset> --use-selector
```

Minimal usage (ShanghaiTech):

```
python main_shanghaitech.py -dp <path to ShanghaiTech dataset> -ee -tt -zl 1024 -ll -use 
```

## Reference
For all the details about the training procedure and the experimental results, please have a look at the [paper](https://arxiv.org/abs/2012.12111).

To cite our work, please use the following form

```
@article{massoli2020mocca,
  title={MOCCA: Multi-Layer One-Class Classification for Anomaly Detection},
  author={Massoli, Fabio Valerio and Falchi, Fabrizio and Kantarci, Alperen and Akti, {\c{S}}eymanur and Ekenel, Hazim Kemal and Amato, Giuseppe},
  journal={arXiv preprint arXiv:2012.12111},
  year={2020}
}
```

## Contacts
If you have any question about our work, please contact [Dr. Fabio Valerio Massoli](mailto:fabio.massoli@isti.cnr.it). 

Have fun! :-D
