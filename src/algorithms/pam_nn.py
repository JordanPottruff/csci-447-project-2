import random

class PamNN:

    def __init__(self, training_data, k):
        self.training_data = training_data
        self.k = k
        self.medoids = self.calculate_medoids(k)

    def calculate_medoids(self, k):
        medoids = random.sample(self.training_data, self.k)
        clusters = []

        last_medoids = None
        while last_medoids is None or medoids != last_medoids:
            # Reset the clusters to have 0 observations.
            clusters = [[] for i in range(self.k)]

            # Assign each observation to the cluster corresponding to the closest medoid.
            for obs in self.training_data.get_data():
                closest_centroid_i = self.find_closest_medoid(obs, medoids)
                clusters[closest_centroid_i].append(obs)

            for cluster_i in range(self.k):
                for obs_i in range(len(self.training_data.get_data())):
                    medoid = medoids[cluster_i]
                    obs = self.training_data.get_data()[cluster_i]
                    if obs in medoids:
                        continue
                    medoids[cluster_i] = obs






        # Placeholder for future value of return statement.
        return ['Attr1Val', 'Attr2Val', 'Attr3Val'], ['Attr1Val', 'Attr2Val', 'Attr3Val']

    # Out of the list of all medoids, this will return the corresponding index for the medoid that is closest to the
    # given observation.
    def find_closest_medoid(self, obs, medoids):
        # Index of the closest centroid seen so far.
        closest_medoid_i = None
        # Distance to the closest centroid seen so far.
        min_dist = None
        for i in range(len(medoids)):
            dist = self.training_data.distance(medoids[i], obs)
            if closest_medoid_i is None or dist < min_dist:
                closest_medoid_i = i
                min_dist = dist
        return closest_medoid_i

    # Calculates the distortion of the given clusters using the training data. Essentially, distortion is a measure of
    # how far each point is from the cluster it belongs to. We use this to optimize our medoid selection.
    def calculate_distortion(self, clusters, medoids):
        distortion = 0
        for cluster_i in range(len(clusters)):
            for obs in clusters[cluster_i]:
                medoid = medoids[cluster_i]
                distortion += self.training_data.distance(obs, medoid)
        return distortion

    def run(self, example):
        # Placeholder for future return value.
        return {'first_class': 1 / 5, 'second_class': 2 / 5, 'third_class': 2 / 5}