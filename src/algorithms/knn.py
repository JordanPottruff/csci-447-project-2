import math
import src.datasets.data_set as ds


class KNN:

    def __init__(self, training_data, k):
        self.training_data = training_data
        self.k = k
        self.distance = self.calc_euclidean_distance()

    def calc_euclidean_distance(self):
        distance = []
        data = self.training_data.get_data()
        for groups in data:
            print(groups)

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
        return distance

    def run(self):
        # Placeholder for future return value.
        return {'first_class': 1/5, 'second_class': 2/5, 'third_class': 2/5}

    def find_k_smallest(self):
        list = [1,23,3,5,6,2,8,23,123,123,67,8,3,222,21]

        k = 4

        # We assign the k_smallest array to start as the first k elements.
        k_smallest = list[0:k]
        # This should always be the largest item in k_smallest (NOT the original list).
        largest = max(k_smallest)

        # These are the items we will then iterate over to find the final k smallest elements.
        left_over = list[k:]

        for item in left_over:
            # This check reduces time because we don't have to check over the k_smallest unless we know we need to
            # update it.
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


