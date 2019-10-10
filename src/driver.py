
import src.util as util
import src.loss as loss
import src.datasets.data_set as ds
import src.algorithms.k_means as kmeans
import src.algorithms.pam_nn as pamnn
import src.algorithms.knn as k_nn
import src.algorithms.edited_knn as e_knn
import src.algorithms.condensed_knn as c_knn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def run_classification(alg_class, data_set, k_values):
    print("-----------------------------------------")
    print("CLASSIFICATION USING " + alg_class.__name__)
    print("Data: " + data_set.filename)
    print(" * N = " + str(len(data_set.data)))

    folds = data_set.validation_folds(10)
    training_data_sizes = []
    accuracies = []
    hinge_losses = []
    for i, k in enumerate(k_values):
        print("[" + str(i+1) + "] k=" + str(k) + " using 10-fold CV")
        avg_accuracy = 0
        avg_hinge_loss = 0
        avg_training_size = 0
        print(" * Folds Complete: ", end='', flush=True)
        for fold_i, fold in enumerate(folds):
            test = fold['test']
            train = fold['train']
            alg = alg_class(train, k)

            results = []
            for obs in test.data:
                result = {"expected": obs[data_set.class_col], "actual": alg.run(obs)}
                results.append(result)

            accuracy = loss.calc_accuracy(results)
            hinge_loss = loss.calc_hinge(results)
            avg_accuracy += accuracy / len(folds)
            avg_hinge_loss += hinge_loss / len(folds)
            avg_training_size += len(alg.training_data.data) / len(folds)
            print(fold_i+1, end='', flush=True)
            if fold_i == len(folds)-1:
                print()
            else:
                print(", ", end='', flush=True)
        avg_accuracy = float("{:.2f}".format(avg_accuracy))
        avg_hinge_loss = float("{:.2f}".format(avg_hinge_loss))
        accuracies.append(avg_accuracy)
        hinge_losses.append(avg_hinge_loss)
        training_data_sizes.append(avg_training_size)
        print(" * Results: ")
        print("   - Avg accuracy = " + str(avg_accuracy))
        print("   - Avg hinge loss = " + str(avg_hinge_loss))
        print()

    return {"k-values": k_values, "losses": [accuracies, hinge_losses], "data-sizes": training_data_sizes}


def run_regression(alg_class, data_set, k_values):
    print("-----------------------------------------")
    print("REGRESSION USING " + alg_class.__name__)
    print("Data: " + data_set.filename)
    print(" * N = " + str(len(data_set.data)))

    folds = data_set.validation_folds(10)
    training_data_sizes = []
    rmse_losses = []
    huber_losses = []
    for i, k in enumerate(k_values):
        print("[" + str(i+1) + "] k=" + str(k) + " using 10-fold CV")
        avg_rmse = 0
        avg_huber_loss = 0
        avg_training_size = 0
        print(" * Folds Complete: ", end='', flush=True)
        for fold_i, fold in enumerate(folds):
            test = fold['test']
            train = fold['train']
            alg = alg_class(train, k)

            results = []
            for obs in test.data:
                result = {"expected": obs[data_set.class_col], "actual": alg.run(obs)}
                results.append(result)

            rmse = loss.calc_rmse(results)
            huber_loss = loss.calc_huber_loss(results)
            avg_rmse += rmse / len(folds)
            avg_huber_loss += huber_loss / len(folds)
            avg_training_size += len(alg.training_data.data) / len(folds)
            print(fold_i + 1, end='', flush=True)
            if fold_i == len(folds) - 1:
                print()
            else:
                print(", ", end='', flush=True)
        avg_rmse = float("{:.2f}".format(avg_rmse))
        avg_huber_loss = float("{:.2f}".format(avg_huber_loss))
        rmse_losses.append(avg_rmse)
        huber_losses.append(avg_huber_loss)
        training_data_sizes.append(avg_training_size)
        print(" * Results: ")
        print("   - Avg root mean squared error = " + str(avg_rmse))
        print("   - Avg huber loss = " + str(avg_huber_loss))
        print()

    return {"k-values": k_values, "losses": [rmse_losses, huber_losses], "data-sizes": training_data_sizes}

def create_metric_chart(results):
    """Generates a table comparing the accuracies for each data set. Parameter input is dictionary with a key
    and a value that is a tuple with 2 lists{key: ([],[])}"""
    labels = []

    label_locations = np.arange(len(results["output"]))  # returns evenly spaced values
    width_of_bars = .6/len(results["output"])

    fig, ax = plt.subplots()

    bars = []
    location = 0

    for dataset, accuracy_and_k in results["output"].items():
        # Get labels for bars
        labels.append(dataset)
        # Get Values of Accuracy and Values of K (Accuracy 1st array, k's second)
        accuracy_list = []
        k_value_list = []
        # Accuracy Extract
        for idx in accuracy_and_k[1]:
            accuracy_list.append(idx)
        # K_value Extract
        for idx in accuracy_and_k[0]:
            k_value_list.append(idx)
        if location == 0:
            # Get bars (Location, height, width, label for legend)
            bars.append(ax.bar(location - .2, accuracy_list[0], width_of_bars, label="K="+str(k_value_list[0]), align='center', color='blue'))
            bars.append(ax.bar(location, accuracy_list[1], width_of_bars, label="K="+str(k_value_list[1]), align='center', color='orange'))
            bars.append(ax.bar(location + .2, accuracy_list[2], width_of_bars, label="K="+str(k_value_list[2]), align='center', color='green'))
        else:
            bars.append(ax.bar(location - .2, accuracy_list[0], width_of_bars, align='center', color='blue'))
            bars.append(ax.bar(location, accuracy_list[1], width_of_bars, align='center',color='orange'))
            bars.append(ax.bar(location + .2, accuracy_list[2], width_of_bars, align='center', color='green'))
        location += 1

    # Add test for labels, title, and custom x-xis tick labels
    ax.set_title(results["title"])
    ax.set_ylabel(results["loss-type"])
    ax.set_xticks(label_locations)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(bars):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for bar in bars:
            height = bar.get_height()
            #print(height)
            ax.annotate('{}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    print(len(bars))
    for i in range(0, len(bars)-1):
         autolabel(bars[i])

    fig.tight_layout()
    plt.show()


def create_chart_data(title, data_set_names, outputs, loss_name, loss_index):
    data = {'title': title, 'loss-type': loss_name, 'output': {}}
    for i in range(len(data_set_names)):
        data['output'][data_set_names[i]] = [outputs[i]["k-values"], outputs[i]["losses"][loss_index]]
    return data

def main():



    # Classification data sets
    abalone_data = ds.get_abalone_data()
    car_data = ds.get_car_data()
    segmentation_data = ds.get_segmentation_data()
    # Regression data sets
    forest_fires_data = ds.get_forest_fires_data()
    machine_data = ds.get_machine_data()
    wine_data = ds.get_wine_data()

    k_values = [5, 25, 50]

    # KNN - Classification
    knn_abalone = run_classification(k_nn.KNN, abalone_data, k_values)
    knn_car = run_classification(k_nn.KNN, car_data, k_values)
    knn_image = run_classification(k_nn.KNN, segmentation_data, k_values)
    accuracy_knn = create_chart_data("KNN", ["abalone", "car", "image"], [knn_abalone, knn_car, knn_image], "Accuracy", 0)
    hinge_knn = create_chart_data("KNN", ["abalone", "car", "image"], [knn_abalone, knn_car, knn_image], "Hinge Loss", 1)

    # KNN - Regression
    knn_fires = run_regression(k_nn.KNN, forest_fires_data, k_values)
    knn_machine = run_regression(k_nn.KNN, machine_data, k_values)
    knn_wine = run_regression(k_nn.KNN, wine_data, k_values)
    rmse_knn = create_chart_data("KNN", ["forest fires", "machine", "wine"], [knn_fires, knn_machine, knn_wine], "RMSE", 0)
    huber_knn = create_chart_data("KNN", ["forest fires", "machine", "wine"], [knn_fires, knn_machine, knn_wine], "Huber Loss", 0)

    # Edited KNN - Classification
    eknn_abalone = run_classification(e_knn.EditedKNN, abalone_data, k_values)
    eknn_car = run_classification(e_knn.EditedKNN, car_data, k_values)
    eknn_image = run_classification(e_knn.EditedKNN, segmentation_data, k_values)
    accuracy_eknn = create_chart_data("Edited KNN", ["abalone", "car", "image"], [eknn_abalone, eknn_car, eknn_image], "Accuracy", 0)
    hinge_eknn = create_chart_data("Edited KNN", ["abalone", "car", "image"], [eknn_abalone, eknn_car, eknn_image], "Hinge Loss", 1)

    # Condensed KNN - Classification
    cknn_abalone = run_classification(c_knn.CondensedKNN, abalone_data, k_values)
    cknn_car = run_classification(c_knn.CondensedKNN, car_data, k_values)
    cknn_image = run_classification(c_knn.CondensedKNN, segmentation_data, k_values)
    accuracy_cknn = create_chart_data("Condensed KNN", ["abalone", "car", "image"], [cknn_abalone, cknn_car, cknn_image], "Accuracy", 0)
    hinge_cknn = create_chart_data("Condensed KNN", ["abalone", "car", "image"], [cknn_abalone, cknn_car, cknn_image], "Hinge Loss", 1)

    classification_clusters = {"abalone": eknn_abalone["data-sizes"], "car": eknn_car["data-sizes"], "image": eknn_image["data-sizes"]}
    regression_clusters = {"forest fires": len(forest_fires_data.data) / 4, "machine": len(machine_data.data) / 4, "wine": len(wine_data.data) / 4}

    # CLUSTER ESTIMATES
    print("Cluster estimates: ")
    print(classification_clusters)
    print(regression_clusters)
    print()

    # K-Means - Classification
    kmeans_abalone = run_classification(kmeans.KMeans, abalone_data, classification_clusters["abalone"])
    kmeans_car = run_classification(kmeans.KMeans, car_data, classification_clusters["car"])
    kmeans_image = run_classification(kmeans.KMeans, segmentation_data, classification_clusters["image"])
    accuracy_kmeans = create_chart_data("K-Means", ["abalone", "car", "image"], [kmeans_abalone, kmeans_car, kmeans_image], "Accuracy", 0)
    hinge_kmeans = create_chart_data("K-Means", ["abalone", "car", "image"], [kmeans_abalone, kmeans_car, kmeans_image], "Hinge Loss", 1)

    # K-Means - Regression
    kmeans_fires = run_regression(kmeans.KMeans, forest_fires_data, regression_clusters["forest fires"])
    kmeans_machine = run_regression(kmeans.KMeans, machine_data, regression_clusters["machine"])
    kmeans_wine = run_regression(kmeans.KMeans, wine_data, regression_clusters["wine"])
    rmse_kmeans = create_chart_data("K-Means", ["forest fires", "machine", "wine"], [kmeans_fires, kmeans_machine, kmeans_wine], "RMSE", 0)
    huber_kmeans = create_chart_data("K-Means", ["forest fires", "machine", "wine"], [kmeans_fires, kmeans_machine, kmeans_wine], "Huber Loss", 0)

    # PAM - Classification
    pam_abalone = run_classification(pamnn.PamNN, abalone_data, classification_clusters["abalone"])
    pam_car = run_classification(pamnn.PamNN, car_data, classification_clusters["car"])
    pam_image = run_classification(pamnn.PamNN, segmentation_data, classification_clusters["image"])
    accuracy_pam = create_chart_data("K-Means", ["abalone", "car", "image"], [pam_abalone, pam_car, pam_image], "Accuracy", 0)
    hinge_pam = create_chart_data("K-Means", ["abalone", "car", "image"], [pam_abalone, pam_car, pam_image], "Hinge Loss", 1)

    # K-Means - Regression
    pam_fires = run_regression(pamnn.PamNN, forest_fires_data, regression_clusters["forest fires"])
    pam_machine = run_regression(pamnn.PamNN, machine_data, regression_clusters["machine"])
    pam_wine = run_regression(pamnn.PamNN, wine_data, regression_clusters["wine"])
    rmse_pam = create_chart_data("PAM-NN", ["forest fires", "machine", "wine"], [pam_fires, pam_machine, pam_wine], "RMSE", 0)
    huber_pam = create_chart_data("PAM-NN", ["forest fires", "machine", "wine"], [pam_fires, pam_machine, pam_wine], "Huber Loss", 0)

    # Chart creation
    create_metric_chart(accuracy_knn)
    create_metric_chart(hinge_knn)
    create_metric_chart(rmse_knn)
    create_metric_chart(huber_knn)
    create_metric_chart(accuracy_eknn)
    create_metric_chart(hinge_eknn)
    create_metric_chart(accuracy_cknn)
    create_metric_chart(hinge_cknn)
    create_metric_chart(accuracy_kmeans)
    create_metric_chart(hinge_kmeans)
    create_metric_chart(rmse_kmeans)
    create_metric_chart(huber_kmeans)
    create_metric_chart(accuracy_pam)
    create_metric_chart(hinge_pam)
    create_metric_chart(rmse_pam)
    create_metric_chart(huber_pam)


main()
