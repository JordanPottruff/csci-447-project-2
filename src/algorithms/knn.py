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
            self.distance.append(self.training_data.distance(groups, test_point))


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

    def find_k_smallest(self, arr_distance, k):
        k_smallest = list[0:k]
        largest = max(k_smallest)
        left_over = list[k:0]
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


        print("k smallest: ")
        print(k_smallest)


