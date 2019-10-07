from algorithms.knn import KNN  # change to src.algorithms.knn
from datasets.data_set import DataSet
import random


class CondensedKNN(KNN):

    def __init__(self, training_data, k):
        super().__init__(training_data, k)
        self.training_data = training_data  # Get dataset
        self.training_list = training_data.get_data()  # Get list of elements associated with dataset
        # Prepare Condensed Training Data and set to be self.training_data
        self.condensed_training_set = []
        self.randomize_training_list()
        training_data.data = self.condense_training_set
        # We then perform KNN on the new dataset...

    def condense_training_data(self):
        # Loop through the randomized data set
        for i in range(len(self.training_list) - 1):
            # If not initializing the condensed data set
            if len(self.condensed_training_set) > 0:
                # Calculate the closest prototype given which parameter we are looking at
                closest_prototype = calculate__closest_prototype(self.training_list[i])
                # If the element we are looking at is classified equivalently to the closest prototype then we ignore
                if self.training_list[i][training_data.class_col] == closest_prototype[training_data.class_col]:
                    continue
                else:  # If not equivalent then add prototype
                    self.condensed_training_set.append(self.training_data[i])
            else:
                # If first one, assume correctly classified and add input
                self.condensed_training_set.append(self.training_data[i])
        return self.condensed_training_set
    
    def calculate_closest_prototype(self, noncondensed_ele):
        """Finds the closest prototype to this element"""
        
        min_dist_prototype = training_data.distance(noncondensed_ele, self.condensed_training_set[0])  # Initialize min distance
        closest_prototype = self.condensed_training_set[0]
        # Loop through condensed training set to find the closest prototype
        for j in range(1, len(self.condensed_training_set) - 1):
                    distance = training_data.distance(noncondensed_ele, self.condensed_training_set[j])
                    if distance < min_dist_prototype:
                        min_dist_prototype = distance
                        closest_prototype = self.condensed_training_set[j]
        return closest_prototype    
        
    def randomize_training_data(self):
        """Randomize the elements/vectors within the list"""
        random.shuffle(self.training_list)
