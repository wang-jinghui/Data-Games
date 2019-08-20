#!/usr/bin/env python
# coding: utf-8


import os
import gc
import sys
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
import keras
import keras.backend as K
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.optimizers import SGD, Adam, Adagrad
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model





warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


train_images =  pickle.load(open('../features/train_img64.pkl', 'rb'))
test_images = pickle.load(open('../features/test_img64.pkl', 'rb'))

# CNN

def CNN_model(input_shape):
    image_inputs = Input(shape=input_shape)
    h = Conv2D(32, (5, 5), activation='relu', padding='same')(image_inputs)
    h = Conv2D(32, (3, 3), activation='relu', padding='same')(h)
    h = MaxPooling2D((3, 3))(h)
    h = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
    h = MaxPooling2D((2, 2))(h)
    #h = Conv2D(128, (3, 3), activation='relu', padding='same')(h)
    #h = MaxPooling2D((3, 3), strides=2)(h)
    h = Flatten()(h)
    h = Dense(512, activation='relu')(h)
    h = Dropout(0.1)(h)
    h = Dense(256, activation='relu')(h)
    h = Dropout(0.1)(h)
    outputs = Dense(9, activation='softmax')(h)
    model = Model(inputs = image_inputs, outputs = outputs)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                            decay=0.0, amsgrad=False)
    #parallel_model = multi_gpu_model(model, gpus=2)
    #parallel_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
    return model


# # CNN-CV

def cnn_cv(train_images, test_images, n_fold=10):
    train_id = train_images.AreaID.tolist()
    train_cate =  train_images.CategoryID.values
    test_id =  test_images.AreaID.tolist()
    test_data =  np.expand_dims(np.array(test_images.Image.tolist()),axis=3)
    oof_proba = np.zeros((train_images.shape[0], 9))
    pre_proba = np.zeros((len(test_data), 9))
    History = {}
    skf = StratifiedKFold(n_splits=n_fold, random_state=2020)
    for i, (train_index, valid_index) in enumerate(skf.split(oof_proba, train_cate)):
        train_x = np.array(train_images.iloc[train_index, 1].tolist())
        train_x = np.expand_dims(train_x, axis=3)
        train_y = train_cate[train_index]
        valid_x = np.array(train_images.iloc[valid_index, 1].tolist())
        valid_y = train_cate[valid_index]
        valid_x = np.expand_dims(valid_x, axis=3)
        train_y = keras.utils.to_categorical(train_y, num_classes=9)
        valid_y = keras.utils.to_categorical(valid_y, num_classes=9)
        model = CNN_model((100, 100, 1))
        lr_reduce = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
        data_augment = ImageDataGenerator(rotation_range=11, zoom_range=0.11, width_shift_range=0.1,
                                                              height_shift_range=0.11, horizontal_flip=False, vertical_flip=False)
                                                              #preprocessing_function=preprocess_func)
        history = model.fit_generator(data_augment.flow(train_x, train_y, batch_size=512), epochs=30, 
                                      validation_data=(valid_x, valid_y),steps_per_epoch=train_x.shape[0]/512, callbacks=[lr_reduce])
        History[i] = history
        oof_proba[valid_index, : ] = model.predict(valid_x)
        pre_proba += model.predict(test_data, batch_size=2048)/skf.n_splits
        gc.enable()
        del train_x, train_y, valid_x, valid_y, model, data_augment
        gc.collect()
    oof_proba = pd.DataFrame(oof_proba)
    pre_proba = pd.DataFrame(pre_proba)
    oof_proba['AreaID'] = train_id
    pre_proba['AreaID'] = test_id
    pre_proba
    return oof_proba, pre_proba, History

# cross validation
oof_cnn, preds_cnn, history_cnn = cnn_cv(train_images, test_images, 5)

# save result
oof_cnn.to_csv('../features/oof_cnn.csv', index=False)
#preds_cnn.AreaID = preds_cnn.AreaID.apply(lambda x: x.split('.')[0])
preds_cnn.to_csv('../features/pre_cnn.csv', index=False)


