# -*- coding: utf-8 -*-
"""
Spyder Editor


"""
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import pandas as pd
import h5py
FTRAIN ='/home/dsui/Desktop/kaggle/dl/training.csv'
FTEST = 'test.csv'

def load_and_clean_data(cols=None):


    df = read_csv(FTRAIN)

    #转为 np.array
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im,sep=' '))   
    if cols:
        df = df[list(cols) + ['Image']]
    # 扔掉有NA的行
    df = df.dropna()  
    #Stack arrays in sequence vertically (row wise).
    X = np.vstack(df['Image'].values) / 255.
    X = X.astype(np.float32)
    #处理label
    y = df[df.columns[:-1]].values
    y = (y - 48) / 48
    y = y.astype(np.float32)
    
    X,y = shuffle(X,y,random_state=42)
    cols = 1
    rows = 2140
    height = 96
    width = 96
   
    data = np.zeros([rows,cols,height,width],float)
    label = np.zeros([1,1,rows,30],float)
    for i in range(rows):
        tmp = X[i][:].reshape(height,width)
        data[i][0][:][:] = tmp
        tmp = y[i][:]
        label[0][0][i][:] = tmp
        #data.append(X[i])
    #d = np.array(data,dtype=np.float32)
    #la = np.array(label,dtype=np.float32)
    print (data)
    
    with h5py.File('training.h5','w') as f:
        f['data'] = data
        f['label'] = label
 
    
    df = read_csv(FTEST)
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im,sep=' '))   
    X = np.vstack(df['Image'].values) / 255.
    X = X.astype(np.float32)
    data = np.zeros([len(X),1,height,width],float)
   # print len(X)
    for i in range(len(X)):
        tmp = X[i][:].reshape(height,width)
        data[i][0][:][:] = tmp
   # d = np.array(data,dtype=np.float32)
    with h5py.File('test.h5','w') as f:
        f['data'] = data
        f['label'] = np.zeros([1,1,len(X),30],float)
    print len(X)
  #  test = pd.HDFStore('test.h5')
 #   test['data'] = X
    
    
load_and_clean_data(None)
    
#if __name__ == '__main__':
#    load_and_clean_data()