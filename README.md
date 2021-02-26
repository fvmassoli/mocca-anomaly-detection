# MOCCA: Multi-Layer One-Class Classification for Anomaly Detection

This repository contains the code relative to the paper "[MOCCA: Multi-Layer One-Class Classification for Anomaly Detection](https://...)" by Fabio Valerio Massoli (ISTI - CNR), Fabrizio Falchi (ISTI - CNR), Alperen Kantarci (ITU), Şeymanur Akti (ITU), Hazim Kemal Ekenel (ITU), Giuseppe Amato (ISTI - CNR).

It reports a new technique to detect anomalies... 

## Proposed Approach


<p align="center">
<img src=""  alt="" width="600" height="300">
</p>


<p align="center">
<img src="" alt="" width="700" height="300">
</p>

## How to run the code
The current version of the code requires python 3.6 and pytorch ...


Minimal usage (CIFAR10):

```
python3 main_cifar10.py -ptr -tr -tt -zl 128 -nc <normal class> -dp <path to CIFAR10 dataset>
```

Minimal usage (MVTec):

```
python3 main_mvtec.py -ptr -tr -tt -zl 128 -nc <normal class> -dp <path to CIFAR10 dataset> --use-selector
```


## Reference
For all the details about the training procedure and the experimental results, please have a look at the [paper](https:/...).

To cite our work, please use the following form

```
... scholar
```

## Model checkpoints

The checkpoints are relative to models reported in **Table 1** of the paper

|   |  | | |  |
| --- | --- | --- | --- | --- |
| [model](https://drive.google.com...) |  |  |  |  |


## Contacts
If you have any question about our work, please contact [Dr. Fabio Valerio Massoli](mailto:fabio.massoli@isti.cnr.it). 

Have fun! :-D
