#!/usr/bin/env python
# coding: utf-8

import os
import sys
import random
import warnings
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import  accuracy_score

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


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


def get_tfidf(count_df):
    temp = count_df.drop(['AreaID','CategoryID'], axis=1)
    columns = temp.columns
    tfidf = TfidfTransformer().fit_transform(temp.values).toarray()
    tfidf = pd.DataFrame(tfidf, columns=columns)
    tfidf['AreaID'] = count_df['AreaID'].values
    tfidf['CategoryID'] = count_df['CategoryID'].values
    return tfidf


def MLP_model(input_shape):
    inputs = Input(shape=input_shape)
    h = Dense(268, activation='relu')(inputs)
    h = Dropout(0.16)(h)
    h = Dense(536, activation='relu')(h)
    h = Dropout(0.16)(h)
    h = Dense(1072, activation='relu')(h)
    h = Dropout(0.16)(h)
    h = Dense(536, activation='relu')(h)
    h = Dropout(0.16)(h)
    outputs = Dense(9, activation='softmax')(h)
    model = Model(inputs=inputs, outputs=outputs)
    adam = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=None,
                            decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
    return model

# # MLP_CV+XGB_CV

def mlp_cv(df , n_fold=10, input_shape=(37,)):
    data = StandardScaler().fit_transform(df.drop(['AreaID', 'CategoryID'], axis=1))
    train_id = df[df['CategoryID'] != -1].AreaID
    test_id = df[df['CategoryID'] == -1].AreaID
    labels = df[df['CategoryID'] != -1].CategoryID
    train_df = data[: len(train_id), : ]
    test_df = data[len(train_id): , :]
    oof_proba = np.zeros((train_df.shape[0], 9))
    pre_proba = np.zeros((test_df.shape[0], 9))
    print(oof_proba.shape, pre_proba.shape)
    History = {}
    #ros = RandomOverSampler(random_state=0)
    skf = StratifiedKFold(n_splits=n_fold, random_state=2020)
    for i, (train_index, valid_index) in enumerate(skf.split(train_df, labels)):
        train_x, train_y = train_df[train_index, :], labels[train_index]
        valid_x, valid_y = train_df[valid_index, :], labels[valid_index]
        # sample balance
        #res_x, res_y = ros.fit_resample(train_x, train_y)
        # OverSampling over
        #res_y = keras.utils.to_categorical(train_y, num_classes=9)
        train_y = keras.utils.to_categorical(train_y, num_classes=9)
        valid_y = keras.utils.to_categorical(valid_y, num_classes=9)
        model = MLP_model(input_shape)
        lr_reduce = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=10, factor=0.5, min_lr=0.00001)
        history = model.fit(train_x,train_y, batch_size=256, epochs=10, validation_data=(valid_x, valid_y),callbacks=[lr_reduce])
        History[i] = history
        oof_proba[valid_index, : ] = model.predict(valid_x)
        pre_proba += model.predict(test_df)/skf.n_splits
    oof_proba = pd.DataFrame(oof_proba)
    pre_proba = pd.DataFrame(pre_proba)
    oof_proba['AreaID'] = train_id
    pre_proba['AreaID'] = test_id.tolist()
    oof_proba.AreaID = oof_proba.AreaID.apply(lambda x: str(x).zfill(6))
    pre_proba.AreaID = pre_proba.AreaID.apply(lambda x:str(x).zfill(6))
    return oof_proba, pre_proba, History


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
    skf = StratifiedKFold(n_splits=n_fold, random_state=2020)
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
                                                     max_delta_step=0, subsample=0.8, 
                                                    colsample_bytree=0.8, colsample_bylevel=1, colsample_bynode=1,
                                                    reg_alpha=3, reg_lambda=5, scale_pos_weight=1, base_score=0.5,
                                                    random_state=2019)
        model.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric=['merror'], verbose=10,
                        early_stopping_rounds=30)
        oof[valid_index, :] = model.predict_proba(valid_x)
        preds += model.predict_proba(test_df)/skf.n_splits
        print('Fold %d : %f'%(i+1, accuracy_score(valid_y, np.argmax(oof[valid_index], axis=1))))
    print('OOF accuracy_score %f'%accuracy_score(labels, np.argmax(oof,axis=1)))
    oof = pd.DataFrame(oof)
    preds = pd.DataFrame(preds)
    oof['AreaID'] = train_id
    preds['AreaID'] = test_id
    return oof, preds


data_v1 = load_data('../features/', 'v1')
data_v2 = load_data('../features/', 'v2')
data_v3 = load_data('../features/','v3')

tfidf_v1 = get_tfidf(data_v1)
tfidf_v2 = get_tfidf(data_v2)

# cross validation
oof_xgb1, pres_xgb1 = xgb_cv(data_v1, 11)

oof_xgb1.to_csv('../features/oof_xgb1.csv', index=False)
pres_xgb1.to_csv('../features/pre_xgb1.csv', index=False)


oof_xgb2, pres_xgb2 = xgb_cv(data_v2, 11)

oof_xgb2.to_csv('../features/oof_xgb2.csv', index=False)
pres_xgb2.to_csv('../features/pre_xgb2.csv', index=False)



oof_xgb5, pres_xgb5 = xgb_cv(data_v3, 10, 0.09)

oof_xgb5.to_csv('../features/oof_xgb5.csv', index=False)
pres_xgb5.to_csv('../features/pre_xgb5.csv', index=False)


# # MLP
oof_mlp1, preds_mlp1, history_mlp1 = mlp_cv(tfidf_v1, 10)

oof_mlp2, preds_mlp2, history_mlp2 = mlp_cv(tfidf_v2, 10)

oof_mlp1.to_csv('../features/oof_mlp1.csv', index=False)
preds_mlp1.to_csv('../features/pre_mlp1.csv', index=False)

oof_mlp2.to_csv('../features/oof_mlp2.csv', index=False)
preds_mlp2.to_csv('../features/pre_mlp2.csv', index=False)


# # xgb

train_user = pickle.load(open('../features/train_user_features.pkl','rb'))
test_user = pickle.load(open('../features/test_user_features.pkl','rb'))

user_oof, user_pre  = xgb_cv(user_df, 10)
user_df = train_user.append(test_user)

user_oof.to_csv('../features/user_oof.csv', index=False)
user_pre.to_csv('../features/user_pre.csv', index=False)

