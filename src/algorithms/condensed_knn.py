from algorithms.knn import KNN  # change to src.algorithms.knn
from datasets.data_set import DataSet
import random


class CondensedKNN(KNN):

    def __init__(self, training_data, k):
        super().__init__(training_data, k)
        self.training_data = training_data
        # Prepare Condensed Training Data and set to be self.training_data
        self.condensed_training_set = []
        self.randomize_training_data()
        self.training_data = self.condense_training_data()

    def condense_training_data(self):
        # Loop through the randomized data set
        for i in range(len(self.training_data) - 1):
            # If not initializing the condensed data set
            if len(self.condensed_training_set) > 0:
                min_dist_prototype = DataSet.distance(self.training_data[i], self.condensed_training_set[0])  # Initialize min distance
                closest_prototype = self.condensed_training_set[0]
                # Calculate the closest prototype
                for j in range(1, len(self.condensed_training_set) - 1):
                    distance = DataSet.distance(self.training_data[i], self.condensed_training_set[j]) # **ERROR:** Currently unsure why self is needed as a parameter, DataSet not initialized ..
                    if distance < min_dist_prototype:
                        min_dist_prototype = distance
                        closest_prototype = self.condensed_training_set[j]
                # If the element we are looking at is classified equivalently to the closest prototype then we ignore
                if self.training_data[i][DataSet.class_col] == self.condensed_training_set[j][DataSet.class_col]:
                    continue
                else:  # If not equivalent then add prototype
                    self.condensed_training_set.append(self.training_data[i])
            else:
                # If first one, assume correctly classified and add input
                self.condensed_training_set.append(self.training_data[i])
        return self.condensed_training_set

    def randomize_training_data(self):
        """Randomize the elements/vectors within the list"""
        random.shuffle(self.training_data)
