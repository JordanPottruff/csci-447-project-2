import src.util as util
import src.datasets.data_set as ds
from src.algorithms.knn import KNN
from src.algorithms.edited_knn import EditedKNN
from src.algorithms.condensed_knn import CondensedKNN
from src.algorithms.k_means import KMeans
from src.algorithms.pam_nn import PamNN

THREE_CLUSTERS_DATA_FILE = "../data/test/three_clusters.data"


def test_knn():
    print("TESTING KNN")
    print("--------------------------------------------------")
    data = get_three_clusters_data()

    # KNN(training data, k-value)
    knn = KNN(data, 5)

    test_near_A = [5, 5, 'A']
    print("Testing: " + str(test_near_A))
    print(knn.run(test_near_A))
    print("5 Nearest Neighbors: ")
    for neighbor in knn.last_nearest_neighbors:
        print(" * {:.2f}".format(neighbor[0]), neighbor[1])
    print()

    test_near_B = [5, 1000, 'B']
    print("Testing: " + str(test_near_B))
    print(knn.run(test_near_B))
    print("5 Nearest Neighbors: ")
    for neighbor in knn.last_nearest_neighbors:
        print(" * {:.2f}".format(neighbor[0]), neighbor[1])
    print()

    test_middle = [30, 28, "UNKNOWN"]
    print("Testing: " + str(test_middle))
    print(knn.run(test_middle))
    print("5 Nearest Neighbors: ")
    for neighbor in knn.last_nearest_neighbors:
        print(" * {:.2f}".format(neighbor[0]), neighbor[1])
    print()


def test_edited_knn():
    print("TESTING EDITED KNN")
    print("--------------------------------------------------")
    data = get_three_clusters_data()

    print("K-value = 1")
    print("Reducing training set...")
    eknn1 = EditedKNN(data, 1)
    print("Reduced training set: ")
    eknn1.training_data.print()

    print("K-value = 5")
    print("Reducing training set...")
    eknn5 = EditedKNN(data, 5)
    print("Reduced training set: ")
    print("N = " + str(len(eknn5.training_data.data)))
    eknn5.training_data.print()
    test_point = [5, 5, "A"]
    print("Testing: " + str(test_point))
    print(eknn5.run(test_point))
    print("5 Nearest Neighbors:")
    for neighbor in eknn5.last_nearest_neighbors:
        if neighbor is not None:
            print(" * {:.2f}".format(neighbor[0]), neighbor[1])
        else:
            print(" * N/A")
    print()


def test_condensed_knn():
    print("TESTING CONDENSED KNN")
    print("--------------------------------------------------")
    data = get_three_clusters_data()

    print("K-value = 3")
    print("Creating reduced training set...")
    cknn5 = CondensedKNN(data, 3)
    print("Reduced training set: ")
    print("N = " + str(len(cknn5.training_data.data)))
    cknn5.training_data.print()
    test_point = [5, 5, "A"]
    print("Testing: " + str(test_point))
    print(cknn5.run(test_point))
    print("3 Nearest Neighbors:")
    for neighbor in cknn5.last_nearest_neighbors:
        if neighbor is not None:
            print(" * {:.2f}".format(neighbor[0]), neighbor[1])
        else:
            print(" * N/A")
    print()


def test_k_means():
    print("TESTING K-MEANS")
    print("--------------------------------------------------")
    data = get_three_clusters_data()

    print("K-value = 3")
    kmeans = KMeans(data, 3)
    print("FINAL Clusters: ")
    for i, centroid in enumerate(kmeans.centroids):
        print(" * " + str(centroid))
        for example in kmeans.clusters[i]:
            print("   - " + str(example))
    print()
    test_point = [60, 5, "C"]
    print("Testing: " + str(test_point))
    print(kmeans.run(test_point))
    print()


def test_pam_nn():
    print("TESTING PAM-NN")
    print("--------------------------------------------------")
    data = get_three_clusters_data()

    print("K-value = 3")
    pam = PamNN(data, 3)
    print("FINAL Clusters: ")
    for i, medoid in enumerate(pam.medoids):
        print(" * " + str(medoid))
        for example in pam.clusters[i]:
            print("   - " + str(example))
    print()
    test_point = [60, 5, "C"]
    print("Testing: " + str(test_point))
    print(pam.run(test_point))
    print()


def get_three_clusters_data():
    data = util.read_file(THREE_CLUSTERS_DATA_FILE)
    three_clusters_data = ds.DataSet(data, 2, [0, 1], THREE_CLUSTERS_DATA_FILE)
    three_clusters_data.convert_to_float([0, 1])
    three_clusters_data.shuffle()
    return three_clusters_data


def main():
    test_knn()
    test_edited_knn()
    test_condensed_knn()
    test_k_means()
    test_pam_nn()

main()
