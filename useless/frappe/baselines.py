import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from deepctr.models import DeepFM, xDeepFM, WDL, PNN, DCN
from deepctr import SingleFeat
from data_loader import load_avazu, load_criteo, load_frappe
# from pyfm import pylibfm


def deepFM(X_train, y_train, X_test, y_test, sparse_feature_list, dense_feature_list):
    model = DeepFM({"sparse": sparse_feature_list, "dense": dense_feature_list}, final_activation='sigmoid',
                   hidden_size=(8,8))
    model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )

    history = model.fit(X_train, y_train, batch_size=256, epochs=5, verbose=2, validation_split=0.111, )
    pred_ans = model.predict(X_test, batch_size=256)
    
    f = open('results.old.2.txt', 'a+')
    f.write('DeepFM test LogLoss: ' + str(round(log_loss(y_test, pred_ans, eps=1e-7), 4)) + '\n')
    f.write('DeepFM test AUC: ' + str(round(roc_auc_score(y_test, pred_ans), 4)) + '\n')
    f.close()
    #print("DeepFM test LogLoss", round(log_loss(y_test, pred_ans, eps=1e-7), 4))
    #print("DeepFM test AUC", round(roc_auc_score(y_test, pred_ans), 4))


def xdeepFM(X_train, y_train, X_test, y_test, sparse_feature_list, dense_feature_list):
    model = xDeepFM({"sparse": sparse_feature_list, "dense": dense_feature_list}, hidden_size=(8,8),
                    cin_layer_size=(4,4))
    model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )

    history = model.fit(X_train, y_train, batch_size=256, epochs=5, verbose=2, validation_split=0.111, )
    pred_ans = model.predict(X_test, batch_size=256)
    
    f = open('results.old.2.txt', 'a+')
    f.write('xDeepFM test LogLoss: ' + str(round(log_loss(y_test, pred_ans, eps=1e-7), 4)) + '\n')
    f.write('xDeepFM test AUC: ' + str(round(roc_auc_score(y_test, pred_ans), 4)) + '\n')
    f.close()
    #print("xdeepFM test LogLoss", round(log_loss(y_test, pred_ans, eps=1e-7), 4))
    #print("xdeepFM test AUC", round(roc_auc_score(y_test, pred_ans), 4))


def dcn(X_train, y_train, X_test, y_test, sparse_feature_list, dense_feature_list):
    model = DCN({"sparse": sparse_feature_list, "dense": dense_feature_list}, hidden_size=(8,))
    model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )

    history = model.fit(X_train, y_train, batch_size=256, epochs=5, verbose=2, validation_split=0.111, )
    pred_ans = model.predict(X_test, batch_size=256)
    
    f = open('results.old.2.txt', 'a+')
    f.write('DCN test LogLoss: ' + str(round(log_loss(y_test, pred_ans, eps=1e-7), 4)) + '\n')
    f.write('DCN test AUC: ' + str(round(roc_auc_score(y_test, pred_ans), 4)) + '\n')
    f.close()
    #print("DCN test LogLoss", round(log_loss(y_test, pred_ans, eps=1e-7), 4))
    #print("DCN test AUC", round(roc_auc_score(y_test, pred_ans), 4))


def wdl(_X_train, y_train, _X_test, y_test, sparse_feature_list, dense_feature_list):
    X_train = _X_train + _X_train
    X_test = _X_test + _X_test
    deep_feature_dim_dict = {"sparse": sparse_feature_list, "dense": dense_feature_list}
    wide_feature_dim_dict = {"sparse": sparse_feature_list, "dense": dense_feature_list}
    model = WDL(deep_feature_dim_dict=deep_feature_dim_dict, wide_feature_dim_dict=wide_feature_dim_dict,
                hidden_size=(8,8))
    model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )

    history = model.fit(X_train, y_train, batch_size=256, epochs=5, verbose=2, validation_split=0.111, )
    pred_ans = model.predict(X_test, batch_size=256)
    
    f = open('results.old.2.txt', 'a+')
    f.write('WDL test LogLoss: ' + str(round(log_loss(y_test, pred_ans, eps=1e-7), 4)) + '\n')
    f.write('WDL test AUC: ' + str(round(roc_auc_score(y_test, pred_ans), 4)) + '\n')
    f.close()
    #print("WDL test LogLoss", round(log_loss(y_test, pred_ans, eps=1e-7), 4))
    #print("WDL test AUC", round(roc_auc_score(y_test, pred_ans), 4))


def pnn(X_train, y_train, X_test, y_test, sparse_feature_list, dense_feature_list):
    model = PNN({"sparse": sparse_feature_list, "dense": dense_feature_list}, hidden_size=(8,))
    model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )

    history = model.fit(X_train, y_train, batch_size=256, epochs=5, verbose=2, validation_split=0.111, )
    pred_ans = model.predict(X_test, batch_size=256)
    
    f = open('results.txt', 'a+')
    f.write('PNN test LogLoss: ' + str(round(log_loss(y_test, pred_ans, eps=1e-7), 4)) + '\n')
    f.write('PNN test AUC: ' + str(round(roc_auc_score(y_test, pred_ans), 4)) + '\n')
    f.close()
    #print("PNN test LogLoss", round(log_loss(y_test, pred_ans, eps=1e-7), 4))
    #print("PNN test AUC", round(roc_auc_score(y_test, pred_ans), 4))


def logistic_regression(X_train, y_train, X_test, y_test):
    lr = LogisticRegression(solver='sag', max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    f = open('results.txt', 'a+')
    f.write('LR test LogLoss: ' + str(round(log_loss(y_test, y_pred, eps=1e-7), 4)) + '\n')
    f.write('LR test AUC: ' + str(round(roc_auc_score(y_test, y_pred), 4)) + '\n')
    f.close()
    #print("LR test LogLoss", round(log_loss(y_test, y_pred, eps=1e-7), 4))
    #print("LR test AUC", round(roc_auc_score(y_test, y_pred), 4))


def factorization_machines(X_train, y_train, X_test, y_test):
    fm = pylibfm.FM(num_factors=50,
                    num_iter=5,
                    verbose=True,
                    task="classification",
                    initial_learning_rate=0.0001,
                    learning_rate_schedule="optimal")
    fm.fit(X_train, y_train)

    y_pred = fm.predict(X_test)

    f = open('results.txt', 'a+')
    f.write('FM test LogLoss: ' + str(round(log_loss(y_test, y_pred, eps=1e-7), 4)) + '\n')
    f.write('FM test AUC: ' + str(round(roc_auc_score(y_test, y_pred), 4)) + '\n')
    f.close()


if __name__ == '__main__':
    #criteo_path = './dataset/criteo/src/train.txt.original'
    avazu_path = './dataset/avazu/train_30percent.csv'

    #avazu_path = './dataset/avazu/train.sample'
    criteo_path = './dataset/criteo/src/train_30percent.csv'

    # frappe_path = "/Users/zyli/Research/InterpRecSys/frappe/neural_factorization_machine/data/frappe/train.csv"
    X_train, y_train, X_test, y_test, sparse_list, dense_list = load_criteo(criteo_path)
    # X_train, y_train, X_test, y_test, sparse_list, dense_list = load_avazu(avazu_path)
    # X_train, y_train, X_test, y_test, sparse_list, dense_list = load_frappe(frappe_path)
    print(len(X_train), X_train[0].shape, y_train.shape, len(X_test), X_test[0].shape, y_test.shape, len(sparse_list), len(dense_list))
    # deepFM(X_train, y_train, X_test, y_test, sparse_list, dense_list)
    # xdeepFM(X_train, y_train, X_test, y_test, sparse_list, dense_list)
    # dcn(X_train, y_train, X_test, y_test, sparse_list, dense_list)
    # pnn(X_train, y_train, X_test, y_test, sparse_list, dense_list)
    # wdl(X_train, y_train, X_test, y_test, sparse_list, dense_list)
    
    X_train = np.array(X_train).T
    X_test = np.array(X_test).T

    enc = OneHotEncoder(handle_unknown='ignore', categorical_features=[i for i in range(len(sparse_list))], n_values='auto')
    enc.fit(X_train)
    X_test = enc.fit_transform(X_test)
    X_train = enc.transform(X_train)

    print(X_test.shape, type(X_test))
    print(X_train.shape, type(X_train))

    y_train = y_train.reshape((y_train.shape[0]))
    y_test = y_test.reshape((y_test.shape[0]))
    factorization_machines(X_train.tocsr(), y_train, X_test.tocsr(), y_test)
    logistic_regression(X_train.tocsr(), y_train, X_test.tocsr(), y_test)
