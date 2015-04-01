


from PIL import Image
from numpy import*
import os
import csv



def read_data(file_path, image_size,labelInfo):
   # file_name_list = os.listdir(file_path)
    data_height = len(labelInfo[0])
    mat_data = zeros((data_height,image_size))
    for index,file_no in enumerate(labelInfo[0]):
        file_name = file_path+'/'+file_no+'.Bmp'
        vec_data = array(Image.open(file_name).convert('L'))
        #
        mat_data[index,:] = vec_data.reshape((1,image_size))

    return mat_data


def read_table(table_name):
    fd = open(table_name)
    ID_list = []
    Class_list = []
    table = []
    fd.readline()
    for line in fd.readlines():
        data_one_line = line.split(',')
        #print (data_one_line)
        ID = data_one_line[0]
        Class = data_one_line[1].strip()
        #if (Class.isalpha() and len(Class)==1):
            #print (ID)
        ID_list.append(ID)
        Class_list.append(Class)
    table.append(ID_list)
    table.append(Class_list)
    #ID_list.append(Class_list)
    return table


def classify0(inX,dataSet,labels,k):
    # inX: input vector
    # dataSet: 
    dataSetSize = dataSet.shape[0]
    
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    sortedDistIndicies = sqDistances.argsort()
    #print (len(sortedDistIndicies))
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #sortedClassCount = sorted(classCount.iteritems(),key=lambda x:x[1],reverse=True)
    #return sortedClassCount[0][0]
    key = None
    val = 0
    for k in classCount:
        #print (k)
        if classCount[k] > val:
            val = classCount[k]
            key = k
    return key


def analysis(trainLabel_file_name, train_data_path,test_data_path,submission_file_name,image_size,k):
    table_cvs = read_table(trainLabel_file_name)
    labels = array(table_cvs[1][:])
    train_data = read_data(train_data_path,image_size,table_cvs)
    #print (train_data.shape)
    submission_cvs = read_table(submission_file_name)
    #file_name_list = os.listdir(test_data_path)
    test_file_name = submission_cvs[0][:]
    #file_id = []
    predict_word = []
    for file_name in test_file_name:
        file_name = test_data_path +'/'+file_name+'.Bmp'
        inX = array(Image.open(file_name).convert('L'))
        inX = inX.reshape((1,image_size))
       # print (inX.shape)
        #file_id.append(file_name.replace(".Bmp",""))
        word = classify0(inX,train_data,labels,k)
        predict_word.append(word)
    c = csv.writer(open("result.csv", "wb"))
    c.writerow(["ID","Class"])
    for i in range(len(test_file_name)):
        c.writerow([test_file_name[i],predict_word[i]])
        
    

def cross_validation(trainLabel_file_name, train_data_path,image_size,k):
    table_cvs = read_table(trainLabel_file_name)
    labels = array(table_cvs[1][:])
    train_data = read_data(train_data_path,image_size,table_cvs)
    hoRatio = 0.1
    m = train_data.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        result = classify0(train_data[i,:],train_data[numTestVecs:m,:],labels[numTestVecs:m],k)
        #print ("predict:%s, labels:%s\n"%(result,labels[i]))
        if( result != labels[i]):
            errorCount += 1.0
    return errorCount / numTestVecs


def test():
    image_size = 20 * 20
    validation = []
    minimal = 10
    minimal_k = 0
    for k in range(4,1000):     
        ratio =cross_validation("trainLabels.csv","trainResized", 400,k)
        if(ratio < minimal):
            minimal = ratio
            minimal_k = k
        validation.append(ratio)
        print ('k is %d, error_ratio is %f'%(k,ratio))
    #result = sorted(validation.items(),key = lambda validation:validation[1],reverse = False)
    #print ("the optimal k is
    #find the index of the minimal value
    print ('result:  k is %d, error_ratio is %f'%(k,ratio))
    fd = open('result','rw')
    fd.write('result:  k is %d, error_ratio is %f'%(k,ratio))
    fd.close()



if __name__ == '__main__':
    #test()
    analysis("trainLabels.csv", "trainResized","testResized","sampleSubmission.csv",400,10)
 
    

