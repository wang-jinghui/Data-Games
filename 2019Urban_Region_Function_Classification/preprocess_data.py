#!/usr/bin/env python
# coding: utf-8
 
import os
import sys
import gc
import pickle
import warnings
import numpy as np
import pandas as pd
from skimage.io import imread
import matplotlib.pyplot as plt
from datetime import datetime, date
from tqdm import tqdm
from multiprocessing import Pool as ProcessPool
from sklearn.feature_extraction.text import CountVectorizer
warnings.filterwarnings('ignore')



# week and hours
def get_visit_feature_v1(input_path, output_path, flag='train', binary=False):
    file_names = os.listdir(input_path)
    if flag == 'train':
        ids = [x.split('.')[0].split('_')[0] for x in file_names]
        labels = [int(x.split('.')[0].split('_')[1])-1 for x in file_names]
    if flag == 'test':
        ids = [x.split('.')[0] for x in file_names]
    w1,w2,w3,w4,w5,w6,w7 = [],[],[],[],[],[],[]
    vocabulary = [ '00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
                            '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
    for f_name in tqdm(file_names):
        empty_dict ={'01':'','02':'','03':'','04':'','05':'','06':'','00':''}
        temp = pd.read_csv(input_path+f_name, '\t', names=['user_id', 'record'])
        elements = ','.join(temp.record)
        elements = elements.split(',')
        for s in elements:
            empty_dict['0'+datetime(int(s[:4]),int(s[4:6]),int(s[6:8])).strftime('%w')]+=s.split('&')[1]+'|'
        w1.append(empty_dict['01'].replace('|', ' '))
        w2.append(empty_dict['02'].replace('|', ' '))
        w3.append(empty_dict['03'].replace('|', ' '))
        w4.append(empty_dict['04'].replace('|', ' '))
        w5.append(empty_dict['05'].replace('|', ' '))
        w6.append(empty_dict['06'].replace('|', ' '))
        w7.append(empty_dict['00'].replace('|', ' '))
        
    monday_array = CountVectorizer(stop_words=None, vocabulary=vocabulary, binary=binary).fit_transform(w1).toarray()
    tuesday_array = CountVectorizer(stop_words=None, vocabulary=vocabulary, binary=binary).fit_transform(w2).toarray()
    wednesday_array = CountVectorizer(stop_words=None, vocabulary=vocabulary, binary=binary).fit_transform(w3).toarray()
    thursday_array = CountVectorizer(stop_words=None, vocabulary=vocabulary, binary=binary).fit_transform(w4).toarray()
    friday_array = CountVectorizer(stop_words=None, vocabulary=vocabulary, binary=binary).fit_transform(w5).toarray()
    saturday_array = CountVectorizer(stop_words=None, vocabulary=vocabulary, binary=binary).fit_transform(w6).toarray()
    sunday_array = CountVectorizer(stop_words=None, vocabulary=vocabulary, binary=binary).fit_transform(w7).toarray()
    
    monday_df = pd.DataFrame(monday_array, columns=['Mon_'+ss for ss in vocabulary])
    tuesday_df = pd.DataFrame(tuesday_array, columns=['Tue_'+ss for ss in  vocabulary])
    wednesday_df = pd.DataFrame(wednesday_array, columns=['Wed_'+ss for ss in vocabulary])
    thursday_df = pd.DataFrame(thursday_array, columns=['Thu_'+ss for ss in vocabulary])
    friday_df = pd.DataFrame(friday_array, columns=['Fri_'+ss for ss in vocabulary])
    saturday_df = pd.DataFrame(saturday_array, columns=['Sat_'+ss for ss in vocabulary])
    sunday_df = pd.DataFrame(sunday_array, columns=['Sun_'+ss for ss in  vocabulary])
    
    df = pd.concat([monday_df, tuesday_df, wednesday_df, thursday_df, friday_df, saturday_df, sunday_df], axis=1)
    df['AreaID'] = ids
    if flag == 'train':
        df['CategoryID'] = labels
    else :
        df['CategoryID'] = -1
    pickle.dump(df, open(output_path, 'wb'))


#  time feature 2
# help-functions
def weekList(record, mode='%w'): # %w
    weeks = ' '.join(['0'+datetime(int(s[:4]),int(s[4:6]),int(s[6:8])).strftime(mode)  for s in record])
    return weeks

def monthList(record):
    months = ' '.join([s[4:6] for s in record])
    return months

def hourList(record):
    hours = ' '.join([x[9:].replace('|',' ') for x in record])
    return hours

def get_visit_feature_v2(input_path, output_path, flag='train'):
    file_names = os.listdir(input_path)
    if flag == 'train':
        ids = [x.split('.')[0].split('_')[0] for x in file_names]
        labels = [int(x.split('.')[0].split('_')[1])-1 for x in file_names]
    if flag == 'test':
        ids = [x.split('.')[0] for x in file_names]
    weeks , months, hours = [], [], []
    v1=['00','01','02','03','04','05','06']
    #v2=['01','02','03','04','05','06','07','08','09','10','11','12']
    v2 = ['10','11','12','01','02','03']
    v3=['00','01','02','03','04','05','06','07','08','09','10','11','12','13', '14','15','16','17','18','19','20','21','22','23']
    for f_name in tqdm(file_names):
        temp = pd.read_csv(input_path+f_name, '\t', names=['user_id', 'record'])
        temp.record = temp.record.apply(lambda x: x.split(','))
        ws = ' '.join(temp.record.apply(weekList).tolist())
        ms = ' '.join(temp.record.apply(weekList).tolist())
        hs = ' '.join(temp.record.apply(hourList).tolist())
        weeks.append(ws)
        months.append(ms)
        hours.append(hs)
    week_array = CountVectorizer(stop_words=None, vocabulary=v1).fit_transform(weeks).toarray()
    month_array = CountVectorizer(stop_words=None, vocabulary=v2).fit_transform(months).toarray()
    hour_array = CountVectorizer(stop_words=None, vocabulary=v3).fit_transform(hours).toarray()
    
    week_df = pd.DataFrame(week_array , columns=['w_'+ss for ss in v1])
    month_df = pd.DataFrame(month_array, columns=['m_'+ss for ss in  v2])
    hour_df = pd.DataFrame(hour_array, columns=['h_'+ss for ss in v3])
    df = pd.concat([week_df, hour_df, month_df], axis=1)
    df['AreaID'] = ids
    if flag == 'train':
        df['CategoryID'] = labels
    else :
        df['CategoryID'] = -1
    pickle.dump(df, open(output_path, 'wb'))


# time feature 3

def get_visit_feature_v3(input_path, output_path, flag='train'):
    file_names = os.listdir(input_path)
    if flag == 'train':
        ids = [x.split('.')[0].split('_')[0] for x in file_names]
        labels = [int(x.split('.')[0].split('_')[1])-1 for x in file_names]
    if flag == 'test':
        ids = [x.split('.')[0] for x in file_names]
    dates = pd.DataFrame({'date':pd.date_range('20181001','20190331', freq='D')})
    dates = dates.date.dt.strftime('%Y%m%d').tolist()
    docs = []
    for f_name in tqdm(file_names):
        temp = pd.read_csv(input_path+f_name, '\t', names=['user_id', 'record'])
        temp.record = temp.record.apply(lambda x: x.split(','))
        text = temp.record.apply(lambda x: ' '.join([s.split('&')[0] for s in x]))
        text = ' '.join(text.tolist())
        docs.append(text)
    count_vector = CountVectorizer(stop_words=None, vocabulary=dates).fit_transform(docs).toarray()
    df = pd.DataFrame(count_vector , columns=['D_'+ss for ss in dates])
    df['AreaID'] = ids
    if flag == 'train':
        df['CategoryID'] = labels
    else :
        df['CategoryID'] = -1
    pickle.dump(df, open(output_path, 'wb'))


# # user feature

def get_user_feature(input_path, output_path):
    train_paths = [input_path+'train_visit/'+str(i)+'/' for i in range(10)]
    test_paths = [input_path+'test_visit/'+str(i)+'/' for i in range(2)]
    temp_dict = {}
    for path in train_paths:
        file_names = os.listdir(path)
        for f_name in tqdm(file_names):
            temp = pd.read_csv(path+f_name, '\t', names=['user_id', 'record'])
            temp.record = temp.record.apply(lambda x: x.split(','))
            for user_id in temp.user_id:
                if user_id not in temp_dict:
                    temp_dict[user_id] = [f_name.split('.')[0].split('_')[1]]
                else:
                    temp_dict[user_id].append(f_name.split('.')[0].split('_')[1])
    # 构造特征 Train
    vocabulary = ['001', '002','003','004','005','006','007','008','009']
    train_ids = []
    train_labels = []
    train_df = []
    for path in train_paths:
        file_names = os.listdir(path)
        train_ids =train_ids+ [x.split('.')[0].split('_')[0] for x in file_names]
        train_labels =train_labels + [int(x.split('.')[0].split('_')[1])-1 for x in file_names]
        for f_name in tqdm(file_names):
            temp = pd.read_csv(path+f_name, '\t', names=['user_id', 'record'])
            temp['cate'] = temp.user_id.apply(lambda x: temp_dict[x])
            cates = temp.cate.apply(lambda x: ' '.join(x)).tolist()
            array = CountVectorizer(stop_words=None, vocabulary=vocabulary).fit_transform(cates).toarray()
            index = int(f_name.split('.')[0].split('_')[1])-1
            array = array.sum(axis=0)
            array[index] = array[index] - temp.shape[0]
            train_df.append(array)
    train_df = pd.DataFrame(train_df, columns=vocabulary)
    train_df['AreaID'] = train_ids
    train_df['CategoryID'] = train_labels
    pickle.dump(train_df, open(output_path+'train_user_features.pkl', 'wb'))
    gc.enable()
    del train_df
    gc.collect()
    # test 特征
    test_ids = []
    test_df = []
    for path in test_paths:
        file_names = os.listdir(path)
        test_ids =test_ids + [x.split('.')[0] for x in file_names]
        for f_name in tqdm(file_names):
            temp = pd.read_csv(path+f_name, '\t', names=['user_id', 'record'])
            temp['cate'] = temp.user_id.apply(lambda x : temp_dict[x] if x in temp_dict else ['unknow'])
            temp = temp[temp.cate != 'unknow']
            if temp.shape[0] ==0:
                test_df.append(list(np.zeros(9)))
            else:
                cates = temp.cate.apply(lambda x: ' '.join(x)).tolist()
                array = CountVectorizer(stop_words=None, vocabulary=vocabulary).fit_transform(cates).toarray()
                array = list(array.sum(axis=0))
                test_df.append(array)
    test_df = pd.DataFrame(test_df, columns=vocabulary)
    test_df['AreaID'] = test_ids
    test_df['CategoryID'] = -1
    pickle.dump(test_df, open(output_path+'test_user_features.pkl', 'wb'))


# merge image 
def get_image_data(input_path, output_path, flag='train'):
    if flag == 'train':
        ids = []
        labels = []
        images = []
        folder_names = os.listdir(input_path)
        for folder_name in tqdm(folder_names):
            file_names = os.listdir(input_path+folder_name+'/')
            ids = ids + [f_name.split('_')[0] for f_name in file_names]
            labels = labels + [int(x.split('.')[0].split('_')[1])-1 for x in file_names]
            images = images + [imread(input_path+folder_name+'/'+file_name, as_gray=True) for file_name in file_names]
        df = pd.DataFrame({'AreaID': ids, 'Image': images, 'CategoryID':labels}) 
    else :
        file_names = os.listdir(input_path)
        ids = [f_name.split('.')[0] for f_name in file_names]
        images = [imread(input_path+'/'+file_name, as_gray=True) for file_name in file_names]
        df = pd.DataFrame({'AreaID': ids,'Image': images})
    pickle.dump(df, open(output_path, 'wb'))



# save images gray format
get_image_data('../data/train_image/', '../features/train_img64.pkl')

get_image_data('../data/test_image/', '../features/test_img64.pkl', flag='test')

# save visit feature 1
for i in range(10):
    get_visit_feature_v1('../data/train_visit/'+str(i)+'/','../features/train_visit_%s_v1.pkl'%(str(i)), 'train')

for i in range(2):
    get_visit_feature_v1('../data/test_visit/'+str(i)+'/','../features/test_visit_%s_v1.pkl'%(str(i)), 'test')

# save visit feature 2
for i in range(10):
    get_visit_feature_v2('../data/train_visit/'+str(i)+'/','../features/train_visit_%s_v2.pkl'%(str(i)), 'train')

for i in range(2):
    get_visit_feature_v2('../data/test_visit/'+str(i)+'/','../features/test_visit_%s_v2.pkl'%(str(i)), 'test')

# save visit feature 3
for i in range(10):
    get_visit_feature_v3('../data/train_visit/'+str(i)+'/','../features/train_visit_%s_v3.pkl'%(str(i)), 'train')

for i in range(2):
    get_visit_feature_v3('../data/test_visit/'+str(i)+'/','../features/test_visit_%s_v3.pkl'%(str(i)), 'test')

# save user feature
get_user_feature('../data/', '../features/')
