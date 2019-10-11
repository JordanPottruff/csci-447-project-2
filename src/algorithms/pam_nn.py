# pam_nn.py
# Implementation of the PAM algorithm, with additional nearest neighbor functionality.
import random
import src.util as util


# The PAM implementation. This will attempt to find k clusters in the training data, and stores the class distribution
# of points belonging to these clusters in order to perform a nearest neighbor classification later on.
class PamNN:

    # Creates an instance of the PAM algorithm. Clusters and their medoids are calculated upon object creation so that
    # the run method can be used right away.
    def __init__(self, training_data, k):
        self.training_data = training_data.copy()
        self.k = k
        self.medoids, self.clusters, self.cluster_classes, self.distortion = self.calculate_medoids(k)

    # The calculation of medoids, clusters, cluster class distributions, and final distortion using the PAM algorithm.
    def calculate_medoids(self, k):
        # We randomly choose our medoids to begin.
        medoids = random.sample(self.training_data.data, self.k)
        clusters = []

        last_medoids = None
        # The cluster calculation will continue until we have "converged", which is signaled by the medoids being
        # unchanged from one cycle to the next.
        while last_medoids is None or medoids != last_medoids:
            last_medoids = medoids.copy()
            # Reset the clusters to have 0 observations.
            clusters = self.assign_clusters(self.training_data.data, medoids)

            # We maintain the initial distortion for comparison later. In addition, we maintain the best possible swap.
            min_distortion = self.calculate_distortion(clusters, medoids)
            best_swap = None
            # We select a medoid (by index)
            for medoid_i in range(self.k):
                # We then select a cluster (by index) to look through.
                for cluster_i in range(len(clusters)):
                    cluster = clusters[cluster_i]
                    # We look through each example (by index) in the cluster.
                    for i in range(len(cluster)):
                        # Our selected medoid:
                        m = medoids[medoid_i]
                        # Our selected example:
                        x = cluster[i]
                        # Verify that the example is not a medoid
                        if x in medoids:
                            continue
                        # Swap the medoid and observation
                        medoids[medoid_i] = x
                        cluster[i] = m
                        # Now calculate the new distortion resulting from the swap...
                        new_distortion = self.calculate_distortion(clusters, medoids)
                        # We now check if this swap results in a lower distortion than our best previous swap. If so, we
                        # record it.
                        if min_distortion > new_distortion:
                            min_distortion = new_distortion
                            best_swap = (medoid_i, (cluster_i, i))
                        # We now reverse our swap.
                        medoids[medoid_i] = m
                        cluster[i] = x
            # If we have a swap that leads to the lowest distortion, then we perform it for real.
            if best_swap is not None:
                m = medoids[best_swap[0]]
                x = clusters[best_swap[1][0]][best_swap[1][1]]
                medoids[best_swap[0]] = x
                clusters[best_swap[1][0]][best_swap[1][1]] = m
            # If there is no best swap, then we could not find a swap that lowered distortion. We then exit.
            else:
                break
        # We will track the distortion of the final clusters:
        final_distortion = self.calculate_distortion(clusters, medoids)

        # Finally, instead of returning each cluster itself, we are only interested in knowing the class distributions
        # in each cluster.
        cluster_classes = [{} for i in range(len(clusters))]
        for i in range(len(clusters)):
            cluster_classes[i] = util.calculate_class_distribution(clusters[i], self.training_data.class_col)
        return medoids, clusters, cluster_classes, final_distortion

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

    def calculate_one_distortion(self, cluster, medoid):
        distortion = 0
        for obs in cluster:
            distortion += self.training_data.distance(obs, medoid)
        return distortion

    def assign_clusters(self, examples, medoids):
        # Set each cluster to be empty
        clusters = [[] for i in range(self.k)]

        # Assign each observation to the cluster corresponding to the closest medoid.
        for obs in examples:
            # We do not want to assign the examples used as medoids into the clusters.
            if obs not in medoids:
                closest_centroid_i = self.find_closest_medoid(obs, medoids)
                clusters[closest_centroid_i].append(obs)
        return clusters

    # Classifies the given example using the clusters found by PAM. We simply find the nearest cluster to our example
    # point and then classify it according to the probability distribution of that cluster.
    def run(self, example):
        closest_centroid_i = self.find_closest_medoid(example, self.medoids)
        return self.cluster_classes[closest_centroid_i]