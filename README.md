# Interpretable Click-through Rate Prediction through Hierarchical Attention

Author: Zeyu Li <zyli@cs.ucla.edu>


## About
This is the Repo for _Interpretable Click-through Rate Prediction through Hierarchical Attention_.
Please find our paper using the following citation
```text
@inproceedings{han2019all,
  title={Interpretable Click-Through Rate Prediction through Hierarchical Attention},
  author={Li, Zeyu and Cheng, Wei and Chen, Yang and Chen, Haifeng and Wang, Wei},
  booktitle={Proceedings of the Thirteenth ACM International Conference on Web Search and Data Mining},
  year={2020},
  organization={ACM}
}
```

## Run

### Dependencies
Please check `requirements.txt` for dependent packages or run
```bash
$ pip install -r requirements.txt
```

### Preprocess dataset
1. Create following structure in the folder
```text
interhat
|-- data
|   |-- raw
|   |   |-- criteoDAC (put unzipped data)
|   |   |-- avazu (put unzipped data)
|   |-- parse
|-- interhat
    |-- ...
```
2. Run preprocess
```bash
$ python interhat/preprocess.py [dataset] [n_buckets]
```

### Run `InterHAt`
Run `run.sh` as an example.

## Datasets
This section introduces the datasets.
### Criteo
```
$ curl -O http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_{'seq -s ',' 0 23'}.gz
```
An useful repo: https://github.com/rambler-digital-solutions/criteo-1tb-benchmark#task-and-data

### Avazu
Avazu dataset is from Kaggle: https://www.kaggle.com/c/avazu-ctr-prediction

### Frappe
Frappe: https://github.com/hexiangnan/neural_factorization_machine/tree/master/data/frappe
