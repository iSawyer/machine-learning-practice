# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 20:28:53 2015

@author: dsui
"""

#import caffe

import numpy as np
caffe_root = "/home/dsui/caffe-master/"
from pandas.io.parsers import read_csv
import sys
import h5py

sys.path.insert(0,caffe_root+'python')

import caffe

MODEL_FILE = 'test.prototxt'
PRETRAINED = '_iter_1000.caffemodel'
FTEST = 'test.csv'

height = 96
width = 96

df = read_csv(FTEST)
df['Image'] = df['Image'].apply(lambda im: np.fromstring(im,sep=' '))   
X = np.vstack(df['Image'].values) / 255.
X = X.astype(np.float32)
print len(X)
test_data = np.zeros([len(X),1,height,width],float)
   # print len(X)
for i in range(len(X)):
    tmp = X[i][:].reshape(height,width)
    test_data[i][0][:][:] = tmp

net = caffe.Net(MODEL_FILE,PRETRAINED)
caffe.set_phase_test()
caffe.set_mode_cpu()

label = np.zeros([len(X),1,1,1])

net.set_input_arrays(test_data.astype(np.float32),label.astype(np.float32))

pred = net.forward()

fs = read_csv('IdLookupTable.csv')



#print len(pred['output'][:][:][0])
look_up_table = {"left_eye_center_x":0,"left_eye_center_y":1,"right_eye_center_x":2,"right_eye_center_y":3,"left_eye_inner_corner_x":4,"left_eye_inner_corner_y":5,\
"left_eye_outer_corner_x":6,\
'left_eye_outer_corner_y':7,\
'right_eye_inner_corner_x':8,\
'right_eye_inner_corner_y':9,\
'right_eye_outer_corner_x':10,\
'right_eye_outer_corner_y':11,\
'left_eyebrow_inner_end_x':12,\
'left_eyebrow_inner_end_y':13,\
'left_eyebrow_outer_end_x':14,\
'left_eyebrow_outer_end_y':15,\
'right_eyebrow_inner_end_x':16,\
'right_eyebrow_inner_end_y':17,\
'right_eyebrow_outer_end_x':18,\
'right_eyebrow_outer_end_y':19,\
'nose_tip_x':20,\
'nose_tip_y':21,\
'mouth_left_corner_x':22,\
'mouth_left_corner_y':23,\
'mouth_right_corner_x':24,\
'mouth_right_corner_y':25,\
'mouth_center_top_lip_x':26,\
'mouth_center_top_lip_y':27,\
'mouth_center_bottom_lip_x':28,\
'mouth_center_bottom_lip_y':29,\
}

res_label = np.zeros([len(X)+1,30])
#print pred

for i in range(len(X)):
    for j in range(30):
    #res_label[i][:] = (pred['output'][:][:][i] * 48) + 48
        res_label[i+1][j] = pred['output'][:][i][:][j]*48 + 48
        if (res_label[i+1][j] > 96):
            res_label[i+1][j]=96
#print res_label
print (res_label.shape)
#print (len(fs['RowId']))
for i in range(len(fs['RowId'])):
    
    image_index = fs['ImageId'][i]
    
    #res_label.append[res_label[i][look_up_table[fs['FeatureName'][i]]]]
   # print image_index
  #  print look_up_table[fs['FeatureName'][i]]
    #print res_label[image_index]
    fs['Location'][i] = res_label[image_index][look_up_table[fs['FeatureName'][i]]]

fs.to_csv('test11.csv')

