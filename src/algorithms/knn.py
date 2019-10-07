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