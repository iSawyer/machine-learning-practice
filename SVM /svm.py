# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 10:03:42 2015
SVM 垃圾邮件分类器 
@author: baicai
"""

import scipy.io as sio 
from numpy import*
from scipy import*
import copy 
#  SMO算法
#  ref ==> http://research.microsoft.com/pubs/69644/tr-98-14.pdf

# main procedure
# 1 outer loop while( iter < maxiter)
# 2 inner loop      for each data in dataset
# 3                     if can be optimize
# 4                           choose another data optimize (SMO)



# 定义结构体存储数据和参数
class OptStruct:
    def __init__(self, dataMat, labelMat, C, toler, KTup):
      
        self.X = dataMat
        self.labelMat = labelMat
        self.C = C
        self.tol = toler
        m,n = dataMat.shape
        self.m = m
        
        self.b = 0
        self.alphas = mat(zeros((self.m,1)))

        # 第一列给出是否是有效的标志位，第二列给出E值
        self.eCache = mat(zeros((self.m,2)))
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:],KTup)
            
def kernelTrans(X, A, KTup):
    m,n = X.shape
    K = zeros((m,1))
    if (KTup[0] == 'lin'):
        K = X * A.T
    elif(KTup[0] == 'RBF'):
        for i in range(m):
            delta_Row = X[i,:] - A
            #print delta_Row.shape
            K[i] = dot(delta_Row,delta_Row)
        K = exp(K / (-1*KTup[1]**2))
    return K
            
            

# 计算偏差，用来跟新 alpha:  alpha_j = alpha_j + (y_j * ( Ei - Ej)) / μ
# 其中 μ = K(xi,xi) + K(xj, xj) - 2K(xi,xj) 若μ<=0 在边界取alpha值
# 之后进行clap, 和对alpha的更新

def calEk(Os, k):
    #print Os.alphas
   # print 'hello'
   # print Os.alphas 
    #print Os.labelMat
    #print Os.K
    fxk = float(multiply(Os.alphas,Os.labelMat).T * Os.K[:,k] + Os.b)
   # print "after fxk"
   # print "fxk:" + str(fxk)
    Ek = fxk - float(Os.labelMat[k])
    return Ek


def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(random.uniform(0,m))
    return j
    
# 启发式寻找 alphaJ
def selectJ(i, Os, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    Os.eCache[i] = [1,Ei]
    #print Os.eCache
    validEcacheList = nonzero(Os.eCache[:,0].A)[0]
    if(len(validEcacheList)) > 1:
        #print "in validEcacheList"
        for k in validEcacheList:
            if k == i: 
                continue
            Ek = calEk(Os, k)
            deltaE = abs(Ei - Ek)
            if(deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK,Ej
    else:
        j = selectJrand(i,Os.m)
        Ej = calEk(Os, j)
    return j,Ej

# 更新Ek 和 ecache
def updateEk(Os,k):
    Ek = calEk(Os,k)
    Os.eCache[k] = [1,Ek]

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def inner(i, Os):
    Ei = calEk(Os,i)
    #print "Ei" + str(Ei)
    tmp_Ey = Os.labelMat[i] * Ei
    
    #print "Os.alphas[i]" + str(Os.alphas[i])
    if( (tmp_Ey < -Os.tol ) and (Os.alphas[i] < Os.C ) ) or ((tmp_Ey > Os.tol) and (Os.alphas[i] > 0)):
    
        # 启发寻找 j
        j,Ej = selectJ(i,Os,Ei)
        #print "Ej" + str(Ej)
        alphaIold = Os.alphas[i].copy()
        alphaJold = Os.alphas[j].copy()
        #print "alphaI" + str(alphaIold)
        #print "alphaJ" + str(alphaJold)
        if(Os.labelMat[i] != Os.labelMat[j]):
            
            L = max(0,Os.alphas[j] - Os.alphas[i])
            H = min(Os.C,Os.C + Os.alphas[j] - Os.alphas[i])
           # print "L" + str(L)
            #print "H" + str(H)
        else:
            
            L = max(0,Os.alphas[j] + Os.alphas[i] - Os.C)
           # print "L" + str(L)
            H = min(Os.C,Os.alphas[j] + Os.alphas[i])
           # print "H" + str(H)
        if L == H: 
            print  "L==H"
            return 0
        
        #eta = K(xi,xi) + K(xj, xj) - 2K(xi,xj)
        # modify when we use kernel
        eta = 2.0 * Os.K[i,j].T - Os.K[i,i] - Os.K[j,j]
        if eta >= 0: 
            print "eta >=0"
            #边界取值
            return 0
        # alpha_j = alpha_j - (y_j * ( Ei - Ej)) / eta
        Os.alphas[j] -= Os.labelMat[j] * (Ei - Ej) / eta
        Os.alphas[j] = clipAlpha(Os.alphas[j],H,L)
        updateEk(Os,j)
        
        if (abs(Os.alphas[j] - alphaJold) < 10e-5):
            print "j not moving enouth"
            return 0
        
        # alpha_i = alpha_i + s(alpha_j - alpha_j_new)  
        # s = yi * yj
        Os.alphas[i] += Os.labelMat[i]*Os.labelMat[j] *(alphaJold - Os.alphas[j])
        updateEk(Os,i)
        
        # bi = Ei + yi(alpha_i_new - alpha_i)*K(xi,xi) + yj(alpha_j_new - alpha_j)K(xj,xj) + b
        # bj the same
        
        b1 = Os.b - Ei - Os.labelMat[i] * (Os.alphas[i] - alphaIold) * Os.K[i,:] * Os.K[i,:].T\
        - Os.labelMat[j] * (Os.alphas[j] - alphaJold) * Os.K[i,:] * Os.K[j,:].T
        b2 = Os.b - Ej - Os.labelMat[i] * (Os.alphas[i] - alphaIold) * Os.K[i,:] * Os.K[j,:].T\
        - Os.labelMat[j] * (Os.alphas[j] - alphaJold) * Os.K[j,:] * Os.K[j,:].T
        if (0 < Os.alphas[i]) and (Os.C > Os.alphas[i]):
            Os.b = b1
        elif ( 0 < Os.alphas[j] and Os.C > Os.alphas[j]):
            Os.b = b2
        else:
            Os.b = (b1 + b2) / 2
        return 1
    else: 
        return 0


def SMO(dataMat, label, C, toler, maxIter, KTup = ('lin',0)):
    Os = OptStruct(mat(dataMat),mat(label),C, toler,KTup)
    iter = 0
    entireSet = True
    alpha_Pair_changed = 0
    while(iter < maxIter) and ((alpha_Pair_changed > 0) or (entireSet)):
        alpha_Pair_changed = 0
        if entireSet:
            for i in range(Os.m):
                alpha_Pair_changed += inner(i,Os)
            
                print "fullSet, iter:%d i:%d, pairs changed %d"%(iter, i, alpha_Pair_changed)
            iter += 1
        else:
            nonBoundIs = nonzero((Os.alphas.A > 0) * (Os.alphas.A < C) )[0]
            for i in nonBoundIs:
                alpha_Pair_changed += inner(i,Os)
                print "nonBoundIs, iter:%d i:%d, pairs changed %d"%(iter, i, alpha_Pair_changed)
            iter += 1
        if entireSet:
            entireSet = False
        elif (alpha_Pair_changed ==0 ):
            entireSet = True
        print "iteration number %d"%(iter)
    return Os.b, Os.alphas

#得到模型
def calcWs(alphas,data, label):
    X = mat(data)
    label = mat(label)
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*label[i], X[i,:].T)
    return w




def testRBF(k1 = 10):
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
   
    train_matrix = mat(train_matrix)
    train_matrix = mat(train_matrix,float)
    train_label = mat(train_label)
    train_label = mat(train_label,float)
    test_matrix = mat(test_matrix)
    test_matrix= mat(test_matrix,float)
    test_label = mat(test_label)
    test_label = mat(test_label,float)
    
    
    for i in range(train_label.shape[0]):
        if(train_label[i] == 0):
            train_label[i] = -1.0
        else:
            train_label[i] = 1.0
   
    
    
   # print shape(test_label)
    
    for i in range(test_label.shape[0]):
        if(test_label[i][0] == 0):
            test_label[i][0] = -1.0
        else:
            test_label[i][0] = 1.0
        
    
    b,alphas = SMO(train_matrix,train_label,1,0.001,('RBF',k1))
    
 
    w = calcWs(alphas,train_matrix,train_label)
    errorCount = 0
    pred_label = test_matrix * w
    for i in range(pred_label.shape[0]):
        if(sign(pred_label[i])!=sign(test_label[i])):
            errorCount += 1
    
    
    print "error count :" + str(errorCount) 

def main():
    testRBF()


if __name__ == '__main__':
    main()

"""
  #test_matrix = mat(test_matrix)
    #test_label = mat(test_label)
    #print test_label
   # return 
    #print alphas
    #svInd = nonzero(alphas.A>0)[0]   
    #sVs = test_matrix[svInd]
    #labelSV = test_label[svInd]
    #print "there are %d SV" %shape(sVs)[0]
 m,n = test_matrix.shape
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, test_matrix[i,:],('RBF',k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if(sign(predict)!=sign(test_label[i])):
            errorCount += 1.0
 """   
    


