
import csv
import random
import math

class DataSet:

    def __init__(self, filename, class_col, attr_cols):
        self.data = read_file(filename)
        self.class_col = class_col
        self.attr_cols = attr_cols

    def get_data(self):
        return self.data

    # Used to handle data sets that involve discrete attribute values. The values in the attribute at the specified
    # column are converted using the given map from the original value to the new value. This is purposefully abstract
    # in order for the DataSet class to work with numerous data sets.
    def convert_attribute(self, col, value_map):
        for row in self.data:
            if row[col] in value_map:
                row[col] = value_map[row[col]]

    def convert_to_float(self, cols):
        for line in self.data:
            for i in range(len(line)):
                if i not in cols:
                    continue
                val = line[i]
                if is_float(val):
                    line[i] = float(val)

    def normalize_z_score(self, cols):
        for col in cols:
            # Calculate the mean of the column's data.
            col_sum = 0
            for row in self.data:
                col_sum += row[col]
            mean = col_sum / len(self.data)

            # Calculate the standard deviation of the column's data.
            sum_square_diffs = 0
            for row in self.data:
                sum_square_diffs += (mean - row[col])**2
            standard_deviation = math.sqrt(sum_square_diffs)

            # Replace each column value with it's z-score.
            for row in self.data:
                z_score = (row[col] - mean) / standard_deviation
                row[col] = z_score

    def shuffle(self):
        random.shuffle(self.data)

    def partition(self, first_percentage):
        cutoff = math.floor(first_percentage * len(self.data))

        first = self.data[:cutoff]
        second = self.data[cutoff:]

        return first, second

    def print(self):
        print()
        for row in self.data:
            print(row)
        print()


def read_file(filename):
    with open(filename) as csvfile:
        data = list(csv.reader(csvfile))
    empty_removed = []
    for line in data:
        if line:
            empty_removed.append(line)
    return empty_removed


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False