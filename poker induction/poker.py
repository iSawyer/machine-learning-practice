# -*- coding: utf-8 -*-
"""
Spyder Editor
Poker rule induction
问题： 随机森林分类
http://www.kaggle.com/c/poker-rule-induction
"""

from numpy import*
import csv
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier


def load_train():
    fd = open("train-1.csv")
    fd.readline()
    csv_fd = csv.reader(fd)
   
    label_train = []
    feature_train = []
  
    for line in csv_fd:
        c = 0
        color_feature = [0,0,0,0]
        number_feature = []
        card = [0,0,0,0,0]
        label_train.append(int(line[-1]))
        line.pop()
        for item in line:
            if c == 0:
                color_feature[int(item)-1] += 1
                c += 1
            else:
                number_feature.append(int(item))
                c += 1
            c = c % 2
            # 处理数字 abs(num(i+1) - num(i))
        number_feature = sort(number_feature)
        for i in range(len(number_feature)):
            card[i] = abs(number_feature[i] - number_feature[(i+1)%len(number_feature)]) 
        color_feature.extend(card)
        feature_train.append(color_feature)
    fd.close()
    return feature_train,label_train
    
def load_test():
    fd = open("test-1.csv")
    fd.readline()
    csv_fd = csv.reader(fd)
    feature_test = []
    for line in csv_fd:
        c = 0
        color_feature = [0,0,0,0]
        number_feature = []
        card = [0,0,0,0,0]
        
       # a = line.pop(0)
        line.pop(0)
        
        for item in line:
            
            if c == 0:
                color_feature[int(item)-1] += 1
                c += 1
            else:
                number_feature.append(int(item))
                c += 1
            c = c % 2
            # 处理数字 abs(num(i+1) - num(i))
        number_feature = sort(number_feature)
        for i in range(len(number_feature)):
            card[i] = abs(number_feature[i] - number_feature[(i+1)%len(number_feature)]) 
        color_feature.extend(card)
        feature_test.append(color_feature)
    fd.close()
    return feature_test
    

def pred():
    feature_train,label_train = load_train()
    rdf = RandomForestClassifier(n_estimators=1600)
    #scores = cross_val_score(rdf, feature_train, label_train)
   # print scores.mean()
    print "contruct done\n"
    rdf = rdf.fit(feature_train,label_train)
    
    print "fit done\n"
    feature_test = load_test()
    print "load test done\n"
    writer = csv.writer(file('Submission.csv','wb'))
    writer.writerow(['id','hand'])
    for index in range(len(feature_test)):
        tt = feature_test[index]
        pd_test = rdf.predict(tt)
        writer.writerow([index+1,pd_test[0]])
    
    
pred()
