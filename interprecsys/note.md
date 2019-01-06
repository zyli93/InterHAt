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

