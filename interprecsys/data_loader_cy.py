import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr import SingleFeat


def load_raw(raw_path):
    data = pd.read_csv(raw_path, header=None, sep='\t')

    exceptions = [16, 17, 25, 29, 34]
    
    sparse_feature = [i for i in range(14, 40) if i not in exceptions]
    dense_feature = [i for i in range(1, 14)]

    data[sparse_feature] = data[sparse_feature].fillna('-1', )
    data[dense_feature] = data[dense_feature].fillna(0, )
    target = 0
    
    print(data.shape, data[sparse_feature].shape, data[dense_feature].shape)

    for feat in sparse_feature:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0,1))
    data[dense_feature] = mms.fit_transform(data[dense_feature])

    sparse_feature_list = [SingleFeat('C' + str(feat), data[feat].nunique()) for feat in sparse_feature]
    dense_feature_list = [SingleFeat('I' + str(feat), 0) for feat in dense_feature]    

    print(sparse_feature_list)
    print(dense_feature_list)

    train, test = train_test_split(data, test_size=0.1)
    train_model_input = [train[feat].values for feat in sparse_feature] + \
                        [train[feat].values for feat in dense_feature]
    test_model_input = [test[feat].values for feat in sparse_feature] + \
                       [test[feat].values for feat in dense_feature]

    return train_model_input, train[target].values, test_model_input, test[target].values, sparse_feature_list, dense_feature_list


def preprocess(data, col_exceptions = []):
    #TODO Preprocess data 
    return data


def load_criteo(criteo_path):
    #TODO load standard criteo library
    col_exceptions = [16, 17, 25, 29, 34]
    sparse_feature = [i for i in range(14, 40) if i not in exceptions]
    dense_feature = [i for i in range(1, 14)]
    target = 0

    split_names = ['train', 'test', 'eval']
    data = {}
    for split in split_names:
        split_data = []
        for (dirpath, dirnames, filenames) in os.walk(os.path.join(criteo_path, split)):
            split_data.append(pd.read_csv(os.path.join(dirpath, filenames), header=None, sep='\t'))
        split_data = preprocess(pd.concat(split_data, axis = 0, ignore_index = True)) 
        
    return data

