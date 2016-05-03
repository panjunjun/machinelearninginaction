'''
Created on Sep 16, 2010
Updated on Mar 3, 2016
kNN: k Nearest Neighbors

Input:      input_x: vector to compare to existing data_set (1xN)
            data_set: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
@update: pan_junjun
'''
from numpy import *
import operator
from os import listdir
from settings import debug


def classify0(input_x, data_set, labels, k):
    data_set_size = data_set.shape[0]
    debug("data_set_size=%s" % data_set_size)
    diff_matrix = tile(input_x, (data_set_size, 1)) - data_set
    debug("diff_matrix=%s" % diff_matrix)
    square_diff_matrix = diff_matrix ** 2
    debug("square_diff_matrix=%s" % square_diff_matrix)
    square_distance = square_diff_matrix.sum(axis=1)
    debug("square_distance=%s" % square_distance)
    distances = square_distance ** 0.5
    debug("distances=%s" % distances)
    sorted_dist_index = distances.argsort()
    debug("sorted_dist_index=%s" % sorted_dist_index)
    class_count = {}
    for i in range(k):
        debug("i=%s" % i)
        vote_i_label = labels[sorted_dist_index[i]]
        debug("vote_i_label=%s" % vote_i_label)
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
        debug("class_count[vote_i_label]=%s" % class_count[vote_i_label])
    debug("loop end")
    sorted_class_count = sorted(class_count.iteritems(),
                                key=operator.itemgetter(1), reverse=True)
    debug("sorted_class_count=%s" % sorted_class_count)
    return sorted_class_count[0][0]


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def file2matrix(filename):
    fr = open(filename)
    array_of_lines = fr.readlines()
    number_of_lines = len(array_of_lines)         #get the number of lines in the file
    debug("number_of_lines=%s" % number_of_lines)
    return_matrix = zeros((number_of_lines, 3))        #prepare matrix to return
    debug("return_matrix=%s" % return_matrix)
    class_label_vector = []                       #prepare labels return
    index = 0
    for line in array_of_lines:
        debug("index=%s" % index)
        debug("line=%s" % line)
        line = line.strip()
        list_from_line = line.split('\t')
        return_matrix[index, :] = list_from_line[0: 3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    debug("loop end")
    return return_matrix, class_label_vector


def autoNorm(data_set):
    minVals = data_set.min(0)
    maxVals = data_set.max(0)
    ranges = maxVals - minVals
    normdata_set = zeros(shape(data_set))
    m = data_set.shape[0]
    normdata_set = data_set - tile(minVals, (m,1))
    normdata_set = normdata_set/tile(ranges, (m,1))   #element wise divide
    return normdata_set, ranges, minVals


def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)


def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))


if __name__ == '__main__':
    ####### create_data_set test #######
    #data_set, label = create_data_set()
    #debug(data_set)
    #debug(label)
    ####### classify0 test #######
    #classified = classify0([0.0, 0.5], data_set, label, 3)
    #print classified
    ####### file2matrix test #######
    dating_data_matrix, dating_labels = file2matrix('datingTestSet2.txt')
    #print dating_data_matrix.shape

