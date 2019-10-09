import math
import src.datasets.data_set as ds
import src.util as util


class KNN:

    def __init__(self, training_data, k):
        self.training_data = training_data
        self.k = k

    # Input the example test, output a list of tuples with (distance, class)
    def find_closest_neighbors(self, observation):
        # Initialize k_smallest to be a list of 'None's. After the for loop is finished it should contain k tuples
        # representing the k-nearest neighbors to the observation. The first position is the distance and the second
        # position is the example itself.
        k_smallest = [None for i in range(self.k)]
        # Stores the index of either (1) the first None value or (2) the largest item if no None's are present.
        max_index = 0
        # Iterate across each example in the training data...
        for example in self.training_data.data:
            dist = self.training_data.distance(example, observation)
            # ...filling in any None values in k_smallest or replacing any examples with larger distances.
            if k_smallest[max_index] is None or k_smallest[max_index][0] > dist:
                k_smallest[max_index] = (dist, example)
                # We also need to re-calculate the max_index given the current k_smallest list.
                for i in range(len(k_smallest)):
                    # If a None is encountered, always set it as max_index (we want remove all Nones right away)
                    if k_smallest[i] is None:
                        max_index = i
                        break
                    # Otherwise, try to find the maximum index if there are no None's.
                    if k_smallest[i][0] < k_smallest[max_index][0]:
                        max_index = i
        # Lastly, we want k_smallest to only store the examples themselves, no distances.
        for i in range(len(k_smallest)):
            k_smallest[i] = k_smallest[i][1]
        return k_smallest

    def run(self, example):
        k_closest = self.find_closest_neighbors(example)
        return util.calculate_class_distribution(k_closest, self.training_data.class_col)



