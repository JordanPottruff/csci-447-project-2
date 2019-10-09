import math
import src.datasets.data_set as ds
import src.util as util


class KNN:

    def __init__(self, training_data, k):
        self.training_data = training_data
        self.k = k
        self.distance = []
    # Input the example test, output a list of tuples with (distance, class)
    def calc_euclidean_distance(self, test_point):
        data = self.training_data.get_data()
        attribute_columns = self.training_data.attr_cols
        class_columns = self.training_data.class_col
        for groups in data:
            transformed_array = [groups[i] for i in attribute_columns]
            test_point = [test_point[i] for i in attribute_columns]
            self.distance.append((self.training_data.distance(transformed_array, test_point), groups[class_columns]))
        k_smallest = self.find_k_smallest(self.distance, self.k)
        # print(k_smallest)
        return k_smallest

    def calc_probability(self, distance_array):
        classification = []
        for i in range(len(distance_array)):
            classification.append((distance_array[i][1]))
        # print(classification)
        class_probability = util.count_requency(classification)
        class_length = len(classification)
        for key, value in class_probability.items():
            class_probability[key] = float(value / class_length)
        return class_probability

    def run(self, example):
        distance = self.calc_euclidean_distance(example)
        return self.calc_probability(distance)

    # TODO(alan): update this function to take in a list of tuples rather than just distances. The tuple should be a
    #  pair of each observation with its distance. We then find the k-th tuples with the shortest distance.
    def find_k_smallest(self, distances, k):
        k_smallest = distances[0:k]
        largest = max(k_smallest)
        left_over = distances[k:]
        for item in left_over:
            if item < largest:
                # We find the largest element in k_smallest now...
                max_index = 0
                for i in range(k):
                    if k_smallest[i] > k_smallest[max_index]:
                        max_index = i
                #... and replace that element with the current item.
                k_smallest[max_index] = item
                # this can be optimized more by doing it in the previous loop, but for simplicity we can just use min.
                largest = max(k_smallest)
        return k_smallest


