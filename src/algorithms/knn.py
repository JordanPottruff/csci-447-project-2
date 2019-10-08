import math
import src.datasets.data_set as ds


class KNN:

    def __init__(self, training_data, k):
        self.training_data = training_data
        self.k = k
        self.distance = []

    def calc_euclidean_distance(self, test_point):
        data = self.training_data.get_data()
        attribute_columns = self.training_data.attr_cols
        print(attribute_columns)
        for groups in data:
            transformed_array = [groups[i] for i in attribute_columns]
            test_point = [test_point[i] for i in attribute_columns]
            self.distance.append(self.training_data.distance(transformed_array, test_point))
        k_smallest = self.find_k_smallest(self.distance, self.k)
        print("Original: ")
        print(sorted(self.distance))
        print("K-smallest")
        print(sorted(k_smallest))


        ## Specific to abalone data ----------------
        # class_col = 8
        # attribute_col = list(range(8))
        # sum = 0
        # for groups in training_data:
        #      for feature in points[groups]:
        #           i = 0
        #           sum += (feature[attribute_cols[i] - test_point[attribute_cols[i++])**2
        #           distance.append(math.sqrt(sum))
        # distance.sorted(distance)
        ## get Kth largest value from the distance list.
        return self.distance

    def run(self):
        # Placeholder for future return value.
        return {'first_class': 1/5, 'second_class': 2/5, 'third_class': 2/5}

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


