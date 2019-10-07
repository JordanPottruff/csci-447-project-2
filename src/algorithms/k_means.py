
import random as rand

class KMeans:

    def __init__(self, training_data, k):
        self.training_data = training_data
        self.k = k
        self.centroids, self.clusters = self.calculate_clusters()

    def calculate_clusters(self):
        # First, generate centroids randomly
        centroids = self.generate_random_centroids()
        clusters = []

        # TODO(jordan): will need to remove this n variable and instead evaluate whether the clusters have changed.
        n = 100
        while n > 0:
            # Reset the clusters to have 0 observations.
            clusters = [[] for i in range(self.k)]

            # Assign each observation to the cluster corresponding to the closest centroid.
            for obs in self.training_data.get_data():
                closest_centroid_i = self.find_closest_centroid(obs, centroids)
                clusters[closest_centroid_i].append(obs)

            # With all observations assigned to a cluster, we now recalculate the centroids as the cluster means.
            for i in range(self.k):
                centroids[i] = self.calculate_cluster_mean(clusters[i])

            n -= 1

        return centroids, clusters

    # Creates a list of centroids that are "random." A centroid is really just a vector of attribute values, so we
    # accomplish this by randomly selecting an attribute value from our training data. Returns the centroids as a 2D
    # list.
    def generate_random_centroids(self):
        num_training_data = len(self.training_data.get_data())
        num_cols = len(self.training_data.get_data()[0])
        centroids = [[] for i in range(self.k)]

        # Initialize the means using randomly selected attribute values.
        for i in range(self.k):
            centroids[i] = [None for i in range(num_cols)]
            for attr_col in self.training_data.attr_cols:
                selected_obs = self.training_data.get_data()[rand.randrange(0, num_training_data)]
                centroids[i][attr_col] = selected_obs[attr_col]
        return centroids

    # Out of the list of all centroids, this will return the corresponding index for the centroid that is closest to the
    # given observation.
    def find_closest_centroid(self, obs, centroids):
        # Index of the closest centroid seen so far.
        closest_centroid_i = None
        # Distance to the closest centroid seen so far.
        min_dist = None
        for i in range(len(centroids)):
            dist = self.training_data.distance(centroids[i], obs)
            if closest_centroid_i is None or dist < min_dist:
                closest_centroid_i = i
                min_dist = dist
        return closest_centroid_i

    # Calculates the mean of the cluster. This is tricky because not all attributes are numeric -- some are strings.
    # When we encounter a string attribute, we simply calculate the mode (using a frequency dictionary) and use that
    # as the "mean".
    def calculate_cluster_mean(self, cluster):
        num_cols = len(self.training_data.get_data()[0])
        attr_cols = self.training_data.attr_cols
        str_attr_cols = self.training_data.get_str_attr_cols()

        # The sums and freqs lists are used for calculating the average and mode of numeric and string attributes,
        # respectively. Both have the same number of columns as our observations, but only the columns corresponding
        # to string (freqs[i]) or numeric (sums[i]) will be filled in.
        sums = [0 for i in range(num_cols)]
        freqs = [{} for i in range(num_cols)]

        # This section does one of two things:
        # (1) calculates the sum of numeric attributes (and stores it in the corresponding columns of 'sums'),
        # (2) calculates the frequency of values in string attributes (and stores it in a dictionary in the freqs list).
        for obs in cluster:
            # We iterate over the attribute columns only (to ignore class and unused columns)
            for attr_col in attr_cols:
                # If our attribute column is string-valued...
                if attr_col in str_attr_cols:
                    # ...then update the frequency table by...
                    str_val = obs[attr_col]
                    if str_val in freqs[attr_col]:
                        # ...incrementing an existing count, if present, or
                        freqs[attr_col][str_val] += 1
                    else:
                        # ...setting the count equal to 1 if value not previously seen.
                        freqs[attr_col][str_val] = 1
                else:
                    # Otherwise, if our column is numeric, we just continue building a sum for that column.
                    sums[attr_col] += obs[attr_col]

        # We now calculate the "means", which is an average for numeric columns and the mode for string columns.
        mean = [0 for i in range(num_cols)]
        # We again want to just iterate over the attribute columns instead of all the columns
        for attr_col in attr_cols:
            # If our attribute column is string-valued...
            if attr_col in str_attr_cols:
                # ...we find the most frequent value...
                freq = freqs[attr_col]
                max_attr_val = None
                for attr_val in freq:
                    if max_attr_val is None or freq[attr_val] > freq[max_attr_val]:
                        max_attr_val = attr_val
                # ...and set the mean for this column to be that value.
                mean[attr_col] = max_attr_val
            else:
                # Otherwise, if our column is numeric, we divide the sum by the size of the cluster to get an average.
                mean[attr_col] = sums[attr_col] / len(cluster)

        # Lastly, we return this mean.
        return mean

    def run(self, example):
        # Placeholder for future return value.
        return {'first_class': 1 / 5, 'second_class': 2 / 5, 'third_class': 2 / 5}
