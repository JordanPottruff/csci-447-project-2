import random
import src.util as util


class PamNN:

    def __init__(self, training_data, k):
        self.training_data = training_data.copy()
        self.k = k
        self.medoids, self.clusters, self.cluster_classes, self.distortion = self.calculate_medoids(k)

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

            # We maintain the initial distortion for comparison later.
            # *OPTIMIZATION*: we segment distortion by calculating the distortion within each cluster. This way, we only
            # need to recalculate distortion for the clusters that change.
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
            if best_swap is not None:
                m = medoids[best_swap[0]]
                x = clusters[best_swap[1][0]][best_swap[1][1]]
                medoids[best_swap[0]] = x
                clusters[best_swap[1][0]][best_swap[1][1]] = m
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

    def run(self, example):
        closest_centroid_i = self.find_closest_medoid(example, self.medoids)
        return self.cluster_classes[closest_centroid_i]