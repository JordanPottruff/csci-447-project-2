# condensed_knn.py
# Implementation of condensed KNN algorithm, which we create as a subclass of the KNN class.

from src.algorithms.knn import KNN  # change to src.algorithms.knn
from src.datasets.data_set import DataSet
import random

# Condensed KNN algorithm. Training data is automatically reduced. To classify, use KNN's "run" method.
class CondensedKNN(KNN):

    # Upon creation of the model, the training data is reduced. We can subsequently use the 'run' method from KNN once
    # the object is created.
    def __init__(self, training_data, k):
        super().__init__(training_data, k)
        self.training_data = self.condense_training_data(training_data.copy())

    # Condenses the self.training_data into a smaller version that should hopefully not lose much accuracy.
    def condense_training_data(self, original_data):
        condensed_training_set = []
        training_list = original_data.get_data().copy()
        random.shuffle(training_list)
        prev_training_set = None

        # While our data set is shrinking...
        while prev_training_set is None or len(prev_training_set) != len(condensed_training_set):
            prev_training_set = condensed_training_set
            # Loop through the randomized data set
            for example in training_list:
                # If not initializing the condensed data set
                if len(condensed_training_set) > 0:
                    # Calculate the closest prototype given which parameter we are looking at
                    closest_prototype = self.calculate_closest_prototype(example, condensed_training_set)
                    # If the element we are looking at is classified equivalently to the closest prototype then we ignore
                    if example[original_data.class_col] == closest_prototype[original_data.class_col]:
                        continue
                    else:  # If not equivalent then add prototype
                        condensed_training_set.append(example)
                else:
                    # If first one, assume correctly classified and add input
                    condensed_training_set.append(example)

        condensed_data = DataSet(condensed_training_set, original_data.class_col, original_data.attr_cols)
        return condensed_data
    
    def calculate_closest_prototype(self, noncondensed_ele, condensed_training_set):
        """Finds the closest prototype to this element"""
        min_dist_prototype = self.training_data.distance(noncondensed_ele, condensed_training_set[0])  # Initialize min distance
        closest_prototype = condensed_training_set[0]
        # Loop through condensed training set to find the closest prototype
        for j in range(1, len(condensed_training_set) - 1):
                    distance = self.training_data.distance(noncondensed_ele, condensed_training_set[j])
                    if distance < min_dist_prototype:
                        min_dist_prototype = distance
                        closest_prototype = condensed_training_set[j]
        return closest_prototype    
