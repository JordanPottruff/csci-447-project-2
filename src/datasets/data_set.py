
import random
import math


class DataSet:

    # Constructs a new data set object using a filename. In addition, the column of the class and a list of columns of
    # the attributes must be specified. Note that the underlying 2D list is not reduced to just the attributes and
    # classes. For example, if you run get_data, it will return a 2D list with all of the columns present, even if a
    # column is neither an attribute or class (i.e. an unused column). To ensure that you are working with the right
    # columns, iterate using the attr_cols or class_col field.
    def __init__(self, data, class_col, attr_cols):
        self.data = data
        self.class_col = class_col
        self.attr_cols = attr_cols

    # Returns the data as a 2D list.
    def get_data(self):
        return self.data

    # Returns a list of columns (indices) that are string values, not numeric. The return value of this function will
    # change depending on the usage of the convert_to_float method.
    def get_str_attr_cols(self):
        str_attr_cols = []
        for attr_col in self.attr_cols:
            if isinstance(self.data[0][attr_col], str):
                str_attr_cols.append(attr_col)
        return str_attr_cols

    # Returns the distance between two observations in the data set. Both a and b are observations that can be from the
    # data set or a completely new data point in the same format.
    def distance(self, a, b):
        str_attr_cols = self.get_str_attr_cols()
        sum = 0
        for attr_col in self.attr_cols:
            if attr_col in str_attr_cols:
                sum += 1 if a[attr_col] != b[attr_col] else 0
            else:
                sum += (a[attr_col] - b[attr_col])**2

        return math.sqrt(sum)

    # Removes the first row from the data. Use if there is header information in that row.
    def remove_header(self):
        self.data = self.data[1:]

    # Used to handle data sets that involve discrete attribute values. The values in the attribute at the specified
    # column are converted using the given map from the original value to the new value. This is purposefully abstract
    # in order for the DataSet class to work with numerous data sets.
    def convert_attribute(self, col, value_map):
        for row in self.data:
            if row[col] in value_map:
                row[col] = value_map[row[col]]

    # Converts values in a specified set of columns (represented as indices) to floating point values.
    def convert_to_float(self, cols):
        for line in self.data:
            for i in range(len(line)):
                if i not in cols:
                    continue
                line[i] = float(line[i])

    # Normalizes the values in a specified set of columns (represented as indices) to a z-score. For interpretation, the
    # new value in the data set represents how many standard deviations an attribute value is from the mean of the
    # attribute.
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

    # Shuffles the rows in the data randomly.
    def shuffle(self):
        random.shuffle(self.data)

    # Partitions the data set into two 2D lists. The first_percentage parameter specifies what proportion of
    # observations should fall into the first 2D list.
    def partition(self, first_percentage):
        cutoff = math.floor(first_percentage * len(self.data))

        first = self.data[:cutoff]
        second = self.data[cutoff:]

        return first, second

    # Prints the data set nicely.
    def print(self):
        print()
        for row in self.data:
            print(row)
        print()
