# -*- coding: utf-8 -*-
"""
Spyder Editor
朴素贝叶斯 垃圾邮件分类器 
测试集误分比 = 5%
转化为log likelihood后 1.6%

"""
import scipy.io as sio 
from numpy import*
from scipy import*

# 数据格式 第一行文件名 第二行 documents-word matrix(m,n), m为邮件数，n为特征
# 第三行为word-bag
# 第四行为数据 第一列为label 后面为word-bag index 
# (i,j) ==> 第i个邮件 第j个单词出现的次数

# naive bayes with multinomial event model and Laplace smoothing.
def bayes_train(matrix,label):
   # tup = preprocess(train_file_name)
    #matrix = train_matrix
    #label = train_label
    # 条件概率： p(y|x) = p(x|y) * p(y) / p(x) = p(x1|y) * p(x2|y) * p(x3|y) ... * p(y)
    # 计算1 p(y = 1) = sum(label(y==1,:)) / m
    # 计算2 p(xi|y=1) = (sum(matrix(x==xi,:)) + 1 ) / (sum(label(y==1,:))*n + |v|)
    # 需要估计的参数 pxi|y=1 pxi|y=0 py=1 py=0 
    m,n = matrix.shape
    #print type(label[0][0])
    y_count = 0.0
    for i in range(m):
        if(label[i] == 1):
            y_count += 1.0
    py_1 = y_count / m
    py_0 = 1.0 - py_1
    pxy_1 = zeros((n,1))
    pxy_0 = zeros((n,1))
    # 每一个样本的长度， 用在拉普拉斯平滑
    n_i = sum(matrix,axis=1)
    # 计算特征频率  
    # 可以将矩阵转制进行效率优化  row-c
    for feature in range(n):
        x_k_y0 = 0
        x_k_y1 = 0
        for sample in range(m):
            if(label[sample] == 1):
                x_k_y1 += matrix[sample][feature]
            else:
                x_k_y0 += matrix[sample][feature]
        pxy_1[feature] = (x_k_y1 + 1) 
        pxy_0[feature] = (x_k_y0 + 1) 
    y1_n = 0
    y0_n = 0
    for sample in range(m):
        if(label[sample] == 1):
            y1_n += n_i[sample]
        else:
            y0_n += n_i[sample]
    # 计算条件概率  p(xk|y) event model + 拉普拉斯平滑
    pxy_1 = (pxy_1) / (y1_n + n)
    pxy_0 = (pxy_0) / (y0_n + n)
    # 没有找到python有序容器啊 
    # top-k 问题用堆 ， 为了简便这里直接排序吧
    dic = {}
    
    for i in range(n):
        p = log(pxy_1[i]) - log(pxy_0[i])
        dic[i] = p[0] 
    
    dd = sorted(dic.iteritems(),key=lambda x: x[1], reverse = True)
    #print dd
    fd = open('../MATRIX.TRAIN')
    fd.readline()
    fd.readline()
    words_bag = fd.readline() 
    words_bag = words_bag.strip('\n')
    word_list = words_bag.split(' ')
    #print dd.keys()
    for i in range(5):
        #print l[i]
        print str(i) + ' ' +  word_list[dd[i][0]]
        
                
    return (py_1,py_0,pxy_1,pxy_0)


def bayes_test(test_matrix, test_label, train_matrix, train_label):
    
    py_1,py_0,pxy_1,pxy_0 = bayes_train(train_matrix,train_label)
    m,n = test_matrix.shape
    pre_label = zeros((m,1))
    # 计算概率, 避免数值过小，求log likelihood 连乘改为加
    for i in range(m):
        ly_1 = 0
        ly_0 = 0
        for j in range(n):
            if(test_matrix[i][j] != 0):
                ly_1 += test_matrix[i][j] * log(pxy_1[j])
                ly_0 += test_matrix[i][j] * log(pxy_0[j])
        ly_1 += log(py_1)
        ly_0 += log(py_0)
        if(ly_1 >= ly_0):
            pre_label[i] = 1
        else:
            pre_label[i] = 0
            
    
    error_count = 0
    for i in range(m):
        if(pre_label[i] != test_label[i]):
            error_count += 1.0
    #print error_count + '\n'
    
    print 'error:' + str(error_count / m)
    
    return 
    
    
def main(test_file_name= 'MATRIX.TEST',train_file_name='MATRIX.TRAIN'):
    data_matrix = sio.loadmat('TRAIN-MATRIX.mat')
    train_matrix = data_matrix['matrix']
    data_label = sio.loadmat('TRAIN-LABEL.mat')
    train_label = data_label['category']
    train_label = transpose(train_label)
    data_matrix = sio.loadmat('TEST-MATRIX.mat')
    test_matrix = data_matrix['matrix']
    data_label = sio.loadmat('TEST-LABEL.mat')
    test_label = data_label['category']
    test_label = transpose(test_label)
    bayes_test(test_matrix,test_label,train_matrix,train_label)
    #bayes_test(test_file_name,train_file_name)
if __name__ == "__main__":
    main()
    #matrix = sio.loadmat('TRAIN-MATRIX.mat')
   # main()

    
    
    
    
    
        
            
        
    
            
#useless!! I use matlab to preprocess data format
"""
def preprocess(file_name):
    fd = open(file_name)
    # 名称
    fd.readline()
    # 大小
    dim = fd.readline()
    dim = dim.split(' ')
    dim[-1].strip('\n')
    m = int(dim[0])
    n = int(dim[1])
    print m,n
    matrix = zeros((m,n))
    fd.readline()
    
    # matrix(1 + cumsum(nums(2:2:end - 1)), m) = nums(3:2:end - 1);
    label = zeros((m,1))    
    row_index = 0
    for line in fd.readlines():
        line = line.strip('\n')
        line_numbers = line.split(' ')
        for i in range(len(line_numbers)):
            line_numbers[i] = int(line_numbers[i])
        
        label[row_index] = line_numbers[0]
        for number in line_numbers:
            
            matrix[row_index][number] += 1
        
        row_index += 1
    fd.close()

#        for line_index in range(len(line_words)):
#            if line_index == 0:
#                continue
 #           print int(line_words[line_index])
 #           if(int(line_words[line_index]==1)):
 #               count += 1
 #           label_index = int(line_words[line_index])
 #           matrix[row_index][label_index] += 1
# 3#       row_index += 1
 #   print 'count:',count


 
    
   # print matrix[0][:]
    return matrix,label

"""
        
    
    
    



    
    