
import random as rand

class KMeans:

    def __init__(self, training_data, k):
        self.training_data = training_data
        self.k = k
        self.clusters = [[] for i in range(k)]
        self.centroids = self.calculate_centroids()

    def calculate_centroids(self):
        self.generate_random_centroids()

        n = 100
        while (n > 0):
            self.clusters = [[] for i in range(self.k)]
            for obs in self.training_data.get_data():
                min_centroid_i = None
                min_dist = None
                for i in range(self.k):
                    centroid = self.centroids[i]
                    dist = self.training_data.distance(centroid, obs)
                    if min_centroid_i is None or dist < min_dist:
                        min_centroid_i = i
                        min_dist = dist
                self.clusters[min_centroid_i].append(obs)

            for i in range(self.k):
                self.centroids[i] = self.calculate_cluster_mean(self.clusters[i])

            num_observations_per_cluster = []
            for cluster in self.clusters:
                num_observations_per_cluster.append(len(cluster))
            print("Observations per cluster: ")
            print(num_observations_per_cluster)
            print()
            n -= 1

        return self.centroids

    def generate_random_centroids(self):
        num_training_data = len(self.training_data.get_data())
        num_cols = len(self.training_data.get_data()[0])
        self.centroids = [[] for i in range(self.k)]

        # Initialize the means using randomly selected attribute values.
        for i in range(self.k):
            self.centroids[i] = [None for i in range(num_cols)]
            for attr_col in self.training_data.attr_cols:
                selected_obs = self.training_data.get_data()[rand.randrange(0, num_training_data)]
                self.centroids[i][attr_col] = selected_obs[attr_col]

    def calculate_cluster_mean(self, cluster):
        num_cols = len(self.training_data.get_data()[0])
        attr_cols = self.training_data.attr_cols
        str_attr_cols = self.training_data.get_str_attr_cols()
        n = len(cluster)

        sums = [0 for i in range(num_cols)]
        freqs = [{} for i in range(num_cols)]

        for obs in cluster:
            for col in range(num_cols):
                if col in attr_cols:
                    val = obs[col]
                    if col in str_attr_cols:
                        if val in freqs[col]:
                            freqs[col][val] += 1
                        else:
                            freqs[col][val] = 1
                    else:
                        sums[col] += obs[col]

        centroid = [0 for i in range(num_cols)]
        for i in range(len(attr_cols)):
            if i in attr_cols:
                if i in str_attr_cols:
                    freq = freqs[i]
                    max_attr_val = None
                    for attr_val in freq:
                        if max_attr_val is None or freq[attr_val] > freq[max_attr_val]:
                            max_attr_val = attr_val
                    centroid[i] = max_attr_val
                else:
                    centroid[i] = sums[i] / n
        return centroid

    def run(self, example):
        # Placeholder for future return value.
        return {'first_class': 1 / 5, 'second_class': 2 / 5, 'third_class': 2 / 5}
