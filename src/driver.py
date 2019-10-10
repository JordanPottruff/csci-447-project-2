
import src.util as util
import src.loss as loss
import src.datasets.data_set as ds
import src.algorithms.k_means as kmeans
import src.algorithms.pam_nn as pamnn
import src.algorithms.knn as k_nn
import src.algorithms.edited_knn as e_nn
import src.algorithms.condensed_knn as ck_nn


def run_classification(alg_class, data_set, k_values):
    print("-----------------------------------------")
    print("CLASSIFICATION USING " + alg_class.__name__)
    print("Data: " + data_set.filename)
    print(" * N = " + str(len(data_set.data)))

    folds = data_set.validation_folds(10)
    accuracies = []
    hinge_losses = []
    for i, k in enumerate(k_values):
        print("[" + str(i+1) + "] k=" + str(k) + " using 10-fold CV")
        avg_accuracy = 0
        avg_hinge_loss = 0
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
            print(fold_i+1, end='', flush=True)
            if fold_i == len(folds)-1:
                print()
            else:
                print(", ", end='', flush=True)
            accuracies.append(accuracy)
            hinge_losses.append(hinge_loss)
        print(" * Results: ")
        print("   - Avg accuracy = " + str(avg_accuracy))
        print("   - Avg hinge loss = " + str(avg_hinge_loss))
        print()
    return k_values, accuracies, hinge_losses


def run_regression(alg_class, data_set, k_values):
    print("-----------------------------------------")
    print("REGRESSION USING " + alg_class.__name__)
    print("Data: " + data_set.filename)
    print(" * N = " + str(len(data_set.data)))

    folds = data_set.validation_folds(10)
    rmse_losses = []
    huber_losses = []
    for i, k in enumerate(k_values):
        print("[" + str(i+1) + "] k=" + str(k) + " using 10-fold CV")
        avg_rmse = 0
        avg_huber_loss = 0
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
            print(fold_i + 1, end='', flush=True)
            if fold_i == len(folds) - 1:
                print()
            else:
                print(", ", end='', flush=True)
        print(" * Results: ")
        print("   - Avg root mean squared error = " + str(avg_rmse))
        print("   - Avg huber loss = " + str(avg_huber_loss))
        print()
        rmse_losses.append(avg_rmse)
        huber_losses.append(avg_huber_loss)
    return k_values, rmse_losses, huber_losses

# def compare_accuracy_table(dataset_accuracies):
#     """Generates a table comparing the accuracies for each data set. Parameter input is a dictionary"""
#     labels = []
#
#     label_locations = np.arange(len(dataset_accuracies))  # returns evenly spaced values
#     width_of_bars = .6/len(dataset_accuracies)
#
#     fig, ax = plt.subplots()
#
#     # We have our label and accuracy
#     bars = []
#     location = 0
#     for dataset, accuracy in dataset_accuracies.items():
#         # Get labels for bars
#         labels.append(dataset)
#         # Get bars (Location, height, width, label for legend)
#         bars.append(ax.bar(location, accuracy, width_of_bars, label=dataset, align='center'))
#         location += 1
#
#     # Add test for labels, title, and custom x-xis tick labels
#     ax.set_title("Accuracies of Datasets")
#     ax.set_ylabel("Accuracy %")
#     ax.set_xticks(label_locations)
#     ax.set_xticklabels(labels)
#     ax.legend()
#
#     def autolabel(bars):
#         """Attach a text label above each bar in *rects*, displaying its height."""
#         for bar in bars:
#             height = bar.get_height()
#             #print(height)
#             ax.annotate('{}'.format(height),
#                         xy=(bar.get_x() + bar.get_width() / 2, height),
#                         xytext=(0, 3),  # 3 points vertical offset
#                         textcoords="offset points",
#                         ha='center', va='bottom')
#     print(len(bars))
#     for i in range(0, len(bars)-1):
#          print(autolabel(bars[i]).get_height())
#          autolabel(bars[i])
#
#     fig.tight_layout()
#     plt.show()

def main():
    # Classification data sets
    abalone_data = ds.get_abalone_data()
    car_data = ds.get_car_data()
    segmentation_data = ds.get_segmentation_data()
    # Regression data sets
    forest_fires_data = ds.get_forest_fires_data()
    machine_data = ds.get_machine_data()
    wine_data = ds.get_wine_data()

    # Classification analysis:
    knn_abalone_out = run_classification(k_nn.KNN, abalone_data, [10, 30, 50])
    knn_car_out = run_classification(k_nn.KNN, car_data, [10, 30, 50])
    knn_segmentation_out = run_classification(k_nn.KNN, segmentation_data, [10, 30, 50])

    # Regression analysis:
    knn_machine_out = run_regression(k_nn.KNN, machine_data, [5, 10, 15])
    knn_forest_fire_out = run_regression(k_nn.KNN, forest_fires_data, [5, 10, 15])
    knn_wine_out = run_regression(k_nn.KNN, wine_data, [5, 10, 15])

main()
