
import csv
import random
import math


class DataSet:

    def __init__(self, filename, class_col, cont_attr_cols, disc_attr_cols):
        self.data = self.read_file(filename)
        self.class_col = class_col
        self.cont_attr_cols = cont_attr_cols
        self.disc_attr_cols = disc_attr_cols

    def read_file(self, filename):
        with open(filename) as csvfile:
            data = list(csv.reader(csvfile))
        empty_removed = []
        for line in data:
            if line:
                empty_removed.append(line)
        return empty_removed

    def remove_missing(self):
        # Removes missing entries in-place -- no return needed.
        'empty'

    def get_data(self):
        return self.data

    # Used to handle data sets that involve discrete attribute values. The values in the attribute at the specified
    # column are converted using the given map from the original value to the new value. This is purposefully abstract
    # in order for the DataSet class to work with numerous data sets.
    def convert_attribute(self, col, map):
        # Will convert attribute values in-place -- no return needed.
        'empty'

    def shuffle(self):
        random.shuffle(self.data)

    def partition(self, first_percentage):
        cutoff = math.floor(first_percentage * len(self.data))

        first = self.data[:cutoff]
        second = self.data[cutoff:]

        return first, second
