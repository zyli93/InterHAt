Implementation note:
-------

### Responsibilities of each file

1. `preprocess.py`
- load raw data files
- split to cus and obj features
- split to train and test sets
- save features to pandas and csv
- should generate X files
    - mixed file train: "clk_thru_id, cus_id, obj_id, cus_feats, obj_feats, target"
       - "clk_thru_train.csv" 
    - mixed file test: "clk_thru_id, cus_id, obj_id, cus_feats, obk_feats, target"
       - "clk_thru_test.csv" 
    - cus file: "cus_id, cus_feats"
       - "customer.csv" 
    - obj file: "obj_id, obj_feats"
       - "object.csv" 
- remove missing features
- remove ignore columns

2. `DataLoader.py`
- load features

### NOTE

1. the label column of the data file should be named "target".
2. need to have "cus_id" and "obj_id" columns
3. move negative sampling to `preprocess.py`. keep if clear from `dataloader`

## Ways of evaluation

1. from kaggle: logarithmic loss (smaller is better), 
1. LogLoss (cross entropy)
2. AUC (there's an implementation)
From PNN paper:
    3. Relative information gain RIG=1-NE (NE: normalized cross entropy)
    4. root mean square error


## Baseline models
0. LR H. B. McMahan, G. Holt, D. Sculley et al., “Ad click prediction: a view from the trenches,” in SIGKDD. ACM, 2013, pp. 1222–1230.
1. FM -
2. DNN (what is FNN?) (check this out: https://github.com/wnzhang/deep-ctr) (xDeepFM say this is plain  DNN plain deep neural network)
3. DCN (a.k.a Deep&Cross)-
4. Wide and Deep (https://github.com/zhougr1993/DeepInterestNetwork/tree/master/wide_deep)
5. DeepFM -
6. xDeepFM -
7. PNN - (several versions)



## Dataset
1. Criteo 

from xDeepFM: Bing/Dianping
from DeepFM:
paper used this dataset: DCN


2. Frappe/MovieLens (AFM paper)
Frappe: https://github.com/hexiangnan/neural_factorization_machine/tree/master/data/frappe

"For both datasets, each log is assigned a target of value 1, 
meaning the user has used the app under the context or applied the tag on the movie.
We randomly pair two negative samples with each log and set their target to −1."

3. iPinYou (PNN paper)
https://pan.baidu.com/s/1kTwX2mF#list/path=%2Fipinyou.contest.dataset&parentPath=%2F

4. Avasu (from Kaggle)



## What else in Experiment?
1. Hyper-param impact
    - Activation function
    - Dropout
    - Num. of neurons per layer
    - Depth of network
    - Num. of hidden layers
    - Network shape (from DeepFM, what is that?)
    
2. 

## Other notes
1. Related work could be long.
2. In experiment part: "we conduct experiments to answer the following questions."


## Next Steps:
1. Select dataset

2.a Implement preprocessing for particular datasets => Train, Valid, Test
    AFM: 70% traing, 20% valid, 10% test
2.b Whether they used valid or no

3. Implement AUC and LogLoss in model evaluation

4. Add different implementations of activation/... as mentioned in `what else in experiment?`

4. Run baseline models

5. learning rate goes to 0.001


## Downloading dataset

### Criteo
```
$ curl -O http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_{'seq -s ',' 0 23'}.gz
```
Useful repos
- https://github.com/rambler-digital-solutions/criteo-1tb-benchmark#task-and-data

### iPinYou
For iPinYou, checkout this github
- https://github.com/wnzhang/make-ipinyou-data


### MovieLens



## NOTEs

2. LR:  (FTRL-Proximal, from paper https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf)
- original repo of a python package: https://github.com/alexeygrigorev/libftrl-python
- a use case for the repo above: https://github.com/alexeygrigorev/nips-ad-placement-challenge
I think you can first try to run the second one.


## Run on nvidia-docker
nvidia-docker run -it --rm -v /local2/zyli/PDER:/workspace/PDER -v /local2/zyli/nltk_data:/workspace/nltk_data nvcr.io/nvidia/pytorch:18.03-py3
nvcr.io/nvidia/tensorflow:18.12-py3
nvidia-docker run -it --rm -v /local2/zyli/InterpRecSys:/workspace/InterpRecSys nvcr.io/nvidia/tensorflow:18.12-py3
