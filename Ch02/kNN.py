"""
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
"""
import operator
from os import listdir

from numpy import *

from settings import debug, DEBUG


def classify0(input_x, data_set, labels, k_value):
    data_set_size = data_set.shape[0]
    debug(data_set_size)
    diff_matrix = tile(input_x, (data_set_size, 1)) - data_set
    debug(diff_matrix)
    square_diff_matrix = diff_matrix ** 2
    debug(square_diff_matrix)
    square_distance = square_diff_matrix.sum(axis=1)
    debug(square_distance)
    distances = square_distance ** 0.5
    debug(distances)
    sorted_dist_index = distances.argsort()
    debug(sorted_dist_index)
    class_count = {}
    for i in range(k_value):
        debug(i)
        vote_i_label = labels[sorted_dist_index[i]]
        debug(vote_i_label)
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
        debug(class_count[vote_i_label])
    print("loop end")
    sorted_class_count = sorted(class_count.iteritems(),
        key=operator.itemgetter(1), reverse=True)
    debug(sorted_class_count)
    return sorted_class_count[0][0]


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels


def file2matrix(filename):
    fr = open(filename)
    array_of_lines = fr.readlines()
    number_of_lines = len(array_of_lines)  # get the number of lines in the file
    debug(number_of_lines)
    return_matrix = zeros((number_of_lines, 3))  # prepare matrix to return
    debug(return_matrix)
    class_label_vector = []  # prepare labels return
    index = 0
    for line in array_of_lines:
        debug(index)
        debug(line)
        line = line.strip()
        list_from_line = line.split("\t")
        return_matrix[index, :] = list_from_line[0: 3]
        class_label_vector.append(int(list_from_line[-1]))  # only for the int element
        index += 1
    print("loop end")
    return return_matrix, class_label_vector


def auto_normalization(data_set):
    min_values = data_set.min(0)
    debug(min_values)
    max_values = data_set.max(0)
    debug(max_values)
    ranges = max_values - min_values

    # norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_values, (m, 1))
    norm_data_set = norm_data_set / tile(ranges, (m, 1))  # element wise divide
    return norm_data_set, ranges, min_values


def dating_class_test():
    ho_ratio = 0.50  # hold out 10%
    dating_data_matrix, dating_labels = file2matrix("datingTestSet2.txt")  # load data setfrom file
    norm_matrix, ranges, min_values = auto_normalization(dating_data_matrix)
    m = norm_matrix.shape[0]
    numeric_test_vectors = int(m * ho_ratio)
    error_count = 0.0
    for i in range(numeric_test_vectors):
        classifier_result = classify0(norm_matrix[i, :], norm_matrix[numeric_test_vectors:m, :],
                                      dating_labels[numeric_test_vectors:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print("the total error rate is: %f" % (error_count / float(numeric_test_vectors)))
    print(error_count)


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir("trainingDigits")  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split(".")[0]  # take off .txt
        classNumStr = int(fileStr.split("_")[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector("trainingDigits/%s" % fileNameStr)
    testFileList = listdir("testDigits")  # iterate through the test set
    error_count = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]  # take off .txt
        classNumStr = int(fileStr.split("_")[0])
        vectorUnderTest = img2vector("testDigits/%s" % fileNameStr)
        classifier_result = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifier_result, classNumStr))
        if (classifier_result != classNumStr):
            error_count += 1.0
    print("\nthe total number of errors is: %d" % error_count)
    print("\nthe total error rate is: %f" % (error_count / float(mTest)))


if __name__ == "__main__":
    ####### create_data_set test #######
    # data_set, label = create_data_set()
    # debug(data_set)
    # debug(label)
    ####### classify0 test #######
    # classified = classify0([0.0, 0.5], data_set, label, 3)
    #print classified
    ####### file2matrix test #######
    dating_data_mat, dating_label = file2matrix("datingTestSet2.txt")
    debug("dating_data_mat=%s" % dating_data_mat)
    debug("dating_label[0:20]=%s" % dating_label[0:20])
    if dating_data_mat.any() and dating_label and DEBUG:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(dating_data_mat[:, 2], dating_data_mat[:, 0], 15.0 * array(dating_label), 15.0 * array(dating_label))
        plt.show()

    norm_mat, value_range, min_value = auto_normalization(dating_data_mat)
    debug("norm_mat=%s" % norm_mat)
    debug("value_range=%s" % value_range)
    debug("min_value=%s" % min_value)

    #print dating_data_matrix.shape
