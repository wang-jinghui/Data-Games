#!/usr/bin/env python
# coding: utf-8
import os
import sys
import random
import warnings
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

def load_data(path, version='v1'):
    train_visit = []
    for i in range(10):
        path_a = path+'train_visit_%s_%s.pkl'%(str(i), version)
        train_visit.append(pickle.load(open(path_a, 'rb')))
    train_visit = pd.concat(train_visit, axis=0)
    test_visit = []
    for i in range(2):
        path_b = path+'test_visit_%s_%s.pkl'%(str(i), version)
        test_visit.append(pickle.load(open(path_b, 'rb')))
    test_visit = pd.concat(test_visit, axis=0)
    data_visit = train_visit.append(test_visit)
    return data_visit

def xgb_cv(data, n_fold, lr=0.09):
    train_df = data[data.CategoryID != -1]
    test_df = data[data.CategoryID == -1]
    train_id = train_df.AreaID.tolist()
    test_id = test_df.AreaID.tolist()
    labels = train_df.CategoryID.values
    train_df = train_df.drop(['AreaID','CategoryID'], axis=1).values
    test_df = test_df.drop(['AreaID', 'CategoryID'], axis=1).values
    #fsel = VarianceThreshold(1.5).fit(train_df)
    #train_df = fsel.transform(train_df)
    #test_df = fsel.transform(test_df)
    oof = np.zeros((train_df.shape[0], 9))
    preds = np.zeros((test_df.shape[0], 9))
    skf = StratifiedKFold(n_splits=n_fold, random_state=2019)
    for i, (train_index, valid_index) in enumerate(skf.split(train_df, labels)):
        train_x, train_y = train_df[train_index, :], labels[train_index]
        valid_x, valid_y = train_df[valid_index,:], labels[valid_index]
        # over sample
        #resa_xindex, res_y = ros.fit_resample(np.arange(train_x.shape[0]).reshape(-1, 1), train_y)
        #index = [x[0] for x in resa_xindex]
        #res_x = train_x[index, :]
        # OverSampling over
        model = xgb.XGBClassifier(max_depth=9, learning_rate=lr, n_estimators=10000,
                                                    objective='multi:softmax', booster='gbtree', n_jobs=-1,
                                                     max_delta_step=0, subsample=0.9, 
                                                    colsample_bytree=0.8, colsample_bylevel=0.9, colsample_bynode=0.85,
                                                    reg_alpha=3, reg_lambda=5, scale_pos_weight=1, base_score=0.5,
                                                    random_state=2019)
        model.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric=['merror'], verbose=10,
                        early_stopping_rounds=50)
        oof[valid_index, :] = model.predict_proba(valid_x)
        preds += model.predict_proba(test_df)/skf.n_splits
        print('Fold %d : %f'%(i+1, accuracy_score(valid_y, np.argmax(oof[valid_index], axis=1))))
    print('OOF accuracy_score %f'%accuracy_score(labels, np.argmax(oof,axis=1)))
    oof = pd.DataFrame(oof)
    preds = pd.DataFrame(preds)
    oof['AreaID'] = train_id
    preds['AreaID'] = test_id
    return oof, preds


def make_submission(preds):
    areaID = preds.AreaID
    categoryID = np.argmax(preds.drop('AreaID', axis=1).values, axis=1)+1
    df = pd.DataFrame({'AreaID':areaID, 'CategoryID':categoryID})
    df.CategoryID = df.CategoryID.apply(lambda x: str(x).zfill(3))
    df.AreaID = df.AreaID.apply(lambda x: str(x).zfill(6))
    df = df.sort_values(by='AreaID')
    return df


def add_fakelabel(train_df, test_df, pred_df, threshold):
    fake_df = pd.DataFrame()
    for i in range(8):
        temp = pred_df[pred_df.iloc[:, i] > threshold].index.tolist()
        temp_df = test_df.iloc[temp, :]
        temp_df['CategoryID'] = i
        fake_df = fake_df.append(temp_df)
    new_train = train_df.append(fake_df)
    return new_train


# # load features
oof_xgb1 = pd.read_csv('../features/oof_xgb1.csv')
pre_xgb1 = pd.read_csv('../features/pre_xgb1.csv')
oof_xgb2 = pd.read_csv('../features/oof_xgb2.csv')
pre_xgb2 = pd.read_csv('../features/pre_xgb2.csv')
oof_xgb5 = pd.read_csv('../features/oof_xgb5.csv')
pre_xgb5 = pd.read_csv('../features/pre_xgb5.csv')
oof_mlp1 = pd.read_csv('../features/oof_mlp.csv')
pre_mlp1 = pd.read_csv('../features/pre_mlp.csv')
oof_mlp2 = pd.read_csv('../features/oof_mlp2.csv')
pre_mlp2 = pd.read_csv('../features/pre_mlp2.csv')
oof_cnn = pd.read_csv('../features/oof_cnn.csv')
pre_cnn = pd.read_csv('../features/pre_cnn.csv')
oof_user = pd.read_csv('../features/user_oof.csv')
pre_user = pd.read_csv('../features/user_pre.csv')

# label 
data_v1 = load_data('../features/', 'v1')

label_df = data_v1[data_v1.CategoryID != -1][['AreaID', 'CategoryID']]
label_df.AreaID = label_df.AreaID.astype(int)


# merge 
train = oof_xgb1.merge(oof_xgb2, on='AreaID', how='left')
train = train.merge(oof_xgb5, on='AreaID', how='left')
train = train.merge(oof_mlp1, on='AreaID', how='left')
train = train.merge(oof_mlp2, on='AreaID', how='left')
train = train.merge(oof_cnn, on='AreaID', how='left')
train = train.merge(oof_user, on='AreaID', how='left')


test = pre_xgb1.merge(pre_xgb2, on='AreaID', how='left')
test = test.merge(pre_xgb5, on='AreaID', how='left')
test = test.merge(pre_mlp1, on='AreaID', how='left')
test = test.merge(pre_mlp2, on='AreaID', how='left')
test = test.merge(pre_cnn, on='AreaID', how='left')
test = test.merge(pre_user, on='AreaID',how='left')

# add label
train = train.merge(label_df, on='AreaID', how='left')
test['CategoryID'] = -1
data = train.append(test)


# final train
xgb_oof, xgb_pre , = xgb_cv(data, 4, lr=0.09)


# # save result
result_mlp = make_submission(xgb_pre)
result_mlp.to_csv('../result/xgb_[xgb125mlp12cnnuser].csv', sep ='\t', index=False, header=None)

