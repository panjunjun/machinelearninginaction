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


def line(length=100):
    return '_' * length


def classify0(input_x, data_set, labels, k_value):
    data_set_size = data_set.shape[0]
    debug("data_set_size = %s" % data_set_size)
    diff_matrix = tile(input_x, (data_set_size, 1)) - data_set
    debug("diff_matrix = %s" % diff_matrix)
    square_diff_matrix = diff_matrix ** 2
    debug("square_diff_matrix = %s" % square_diff_matrix)
    square_distance = square_diff_matrix.sum(axis=1)
    debug("square_distance = %s" % square_distance)
    distances = square_distance ** 0.5
    debug("distances = %s" % distances)
    sorted_dist_index = distances.argsort()
    debug("sorted_dist_index = %s" % sorted_dist_index)
    class_count = {}
    for i in range(k_value):
        vote_i_label = labels[sorted_dist_index[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    debug("sorted_class_count = %s" % sorted_class_count)
    return sorted_class_count[0][0]


def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_of_playing_video_games = float(raw_input("percentage of time spent playing vedio games?"))
    frequent_fly_miles = float(raw_input("frequent flier miles earned per year?"))
    ice_cream = float(raw_input("liters of ice cream consumed per year?"))
    dating_data_matrix, dating_labels = file_to_matrix("datingTestSet2.txt")
    norm_matrix, ranges, min_values = auto_normalization(dating_data_matrix)
    input_array = array([frequent_fly_miles, percent_of_playing_video_games, ice_cream])
    classifier_result = classify0((input_array - min_values) / ranges, norm_matrix, dating_labels, 3)
    print("you will probably like this person: %s" % result_list[classifier_result - 1])


def create_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels


def file_to_matrix(filename):
    fr = open(filename)
    array_of_lines = fr.readlines()
    number_of_lines = len(array_of_lines)  # get the number of lines in the file
    debug("number_of_lines = %s" % number_of_lines)
    return_matrix = zeros((number_of_lines, 3))  # prepare matrix to return
    debug("return_matrix = %s" % return_matrix)
    class_label_vector = []  # prepare labels return
    index = 0
    for line in array_of_lines:
        line = line.strip()
        list_from_line = line.split("\t")
        return_matrix[index, :] = list_from_line[0: 3]
        class_label_vector.append(int(list_from_line[-1]))  # only for the int element
        index += 1
    print("loop end")
    return return_matrix, class_label_vector


def auto_normalization(data_set):
    min_value = data_set.min(0)
    debug("min_value = %s" % min_value)
    max_values = data_set.max(0)
    debug("max_values = %s" % max_values)
    ranges = max_values - min_value

    # norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_value, (m, 1)) # tile: copy 1st argument value to 2nd argument shape
    norm_data_set = norm_data_set / tile(ranges, (m, 1))  # element wise divide
    return norm_data_set, ranges, min_value


def dating_class_test():
    debug(line())
    ho_ratio = 0.1  # hold out 10%
    debug("ho_ratio = %s" % ho_ratio)
    dating_data_matrix, dating_labels = file_to_matrix("datingTestSet2.txt")  # load data setfrom file
    debug("dating_data_matrix = %s" % dating_data_matrix)
    debug("dating_labels = %s" % dating_labels)
    norm_matrix, ranges, min_value = auto_normalization(dating_data_matrix)
    debug("norm_matrix = %s" % norm_matrix)
    debug("ranges = %s" % ranges)
    debug("min_value = %s" % min_value)
    m = norm_matrix.shape[0]
    print("norm_matrix.shape is %s" % str(norm_matrix.shape))
    debug("m = %s" % m)
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


def image_to_vector(file_name):
    return_vector = zeros((1, 1024))
    fr = open(file_name)
    for i in range(32):
        line_string = fr.readline()
        for j in range(32):
            return_vector[0, 32 * i + j] = int(line_string[j])
    return return_vector


def hand_writing_classification_test():
    hand_writing_label = []
    training_file_list = listdir("digits/trainingDigits")  # load the training set
    m = len(training_file_list)
    training_matrix = zeros((m, 1024))
    for i in range(m):
        file_name_string = training_file_list[i]
        file_string = file_name_string.split(".")[0]  # take off .txt
        class_number_string = int(file_string.split("_")[0])
        hand_writing_label.append(class_number_string)
        training_matrix[i, :] = image_to_vector("digits/trainingDigits/%s" % file_name_string)
    test_file_list = listdir("digits/testDigits")  # iterate through the test set
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_string = test_file_list[i]
        file_string = file_name_string.split(".")[0]  # take off .txt
        class_number_string = int(file_string.split("_")[0])
        vector_under_test = image_to_vector("digits/testDigits/%s" % file_name_string)
        classifier_result = classify0(vector_under_test, training_matrix, hand_writing_label, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifier_result, class_number_string))
        if classifier_result != class_number_string:
            error_count += 1.0
    print("\nthe total number of errors is: %d" % error_count)
    print("\nthe total error rate is: %f" % (error_count / float(m_test)))


if __name__ == "__main__":
    ####### create_data_set test #######
    data_set_test, label_test = create_data_set()
    print("data_set = %s" % data_set_test)
    print("label = %s" % label_test)
    ####### classify0 test #######
    classification = classify0([0.0, 0.5], data_set_test, label_test, 3)
    print("classification = %s" % classification)
    ####### file_to_matrix test #######
    dating_data_matrix_test, dating_label_test = file_to_matrix("datingTestSet2.txt")
    print("dating_data_matrix_test = %s" % dating_data_matrix_test)
    print("dating_label_test[0:20] = %s" % dating_label_test[0:20])

    # if dating_data_matrix_test.any() and dating_label_test and DEBUG:
    #     import matplotlib.pyplot as plt
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.scatter(dating_data_matrix_test[:, 0], dating_data_matrix_test[:, 1], 15.0 * array(dating_label_test), 15.0 * array(dating_label_test))
    #     plt.show()

    norm_mat_test, value_range_test, min_value_test = auto_normalization(dating_data_matrix_test)
    print("norm_mat_test=%s" % norm_mat_test)
    print("value_range_test=%s" % value_range_test)
    print("min_value_test=%s" % min_value_test)

    # dating_class_test()
    # classify_person()
    hand_writing_classification_test()