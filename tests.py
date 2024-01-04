import os
import numpy as np
from tabulate import tabulate
from scipy.stats import ttest_ind
import constants
from preprocessing.os_helper import create_dir_if_not_exist

datasets = os.listdir(constants.DATASETS_DIR)
pruning_values = constants.PRUNING_VALUES
pruning_methods = ["prune_worst", "prune_random"]

scores_save_path_train = os.path.join(constants.MODELS_DIR, constants.TRAINED_MODELS_TO_PRUNE_CURRENT_DATETIME_DIR, constants.TRAINING_SCORES_NAME)
scores_save_path_prune = os.path.join(constants.PRUNING_DIR, constants.PRUNED_MODELS_TO_TEST_CURRENT_DATETIME_DIR, constants.PRUNING_SCORES_NAME)

trained_scores = np.load(scores_save_path_train)    # scores[dataset_id, fold_id] = f1_result
pruned_scores = np.load(scores_save_path_prune)     # scores[dataset_id, pruning_method_id, pruning_value_id, fold_id]


table_default_avg = tabulate(zip(datasets, np.mean(trained_scores, axis=-1)), tablefmt="grid", headers=["dataset", "F1 score avg"])
print(table_default_avg)
print()
table_default_std = tabulate(zip(datasets, np.std(trained_scores, axis=-1)), tablefmt="grid", headers=["dataset", "F1 score std"])
print(table_default_std)


pruned_folds_mean = np.mean(pruned_scores, axis=1)      # zamienić na -1
# pruned_folds_mean = np.mean(pruned_scores, axis=-1)
table = tabulate(pruned_folds_mean[:, 0, :], tablefmt="grid", headers=pruning_values, showindex=datasets)
print(pruning_methods[0])
print(table)

table = tabulate(pruned_folds_mean[:, 1, :], tablefmt="grid", headers=pruning_values, showindex=datasets)
print(pruning_methods[1])
print(table)


def compare_pruning_methods():
    scores_difference = np.zeros((len(datasets), len(constants.PRUNING_VALUES)))
    scores_p_value = np.zeros((len(datasets), len(constants.PRUNING_VALUES)))
    for dataset_id, dataset in enumerate(datasets):
        for pruning_id, pruning_val in enumerate(constants.PRUNING_VALUES):
            scores_difference[dataset_id, pruning_id], scores_p_value[dataset_id, pruning_id] = ttest_ind(
                pruned_scores[dataset_id, :, 0, pruning_id], pruned_scores[dataset_id, :, 1, pruning_id]
            )
    return scores_difference, scores_p_value


def compare_f1_scores_using_pruning_worst():
    scores_difference = np.zeros((len(datasets), len(constants.PRUNING_VALUES)))
    scores_p_value = np.zeros((len(datasets), len(constants.PRUNING_VALUES)))
    for dataset_id, dataset in enumerate(datasets):
        for pruning_id, pruning_val in enumerate(constants.PRUNING_VALUES):
            scores_difference[dataset_id, pruning_id], scores_p_value[dataset_id, pruning_id] = ttest_ind(
                pruned_scores[dataset_id, :, 0, 0], pruned_scores[dataset_id, :, 0, pruning_id]
            )
    return scores_difference, scores_p_value


def compare_f1_scores_using_pruning_random():
    scores_difference = np.zeros((len(datasets), len(constants.PRUNING_VALUES)))
    scores_p_value = np.zeros((len(datasets), len(constants.PRUNING_VALUES)))
    for dataset_id, dataset in enumerate(datasets):
        for pruning_id, pruning_val in enumerate(constants.PRUNING_VALUES):
            scores_difference[dataset_id, pruning_id], scores_p_value[dataset_id, pruning_id] = ttest_ind(
                pruned_scores[dataset_id, :, 1, 0], pruned_scores[dataset_id, :, 1, pruning_id]
            )
    return scores_difference, scores_p_value


def generate_worse_significant_table(scores_difference, scores_p_value):
    worse = np.empty((len(datasets), len(constants.PRUNING_VALUES)), dtype='|S32')
    worse[:] = "BETTER"
    significant = np.empty((len(datasets), len(constants.PRUNING_VALUES)), dtype='|S32')
    significant[:] = "NOT significant"
    worse[scores_difference >= 0] = "WORSE"
    significant[scores_p_value < constants.P_VALUE] = "SIGNIFICANT"

    for idx, (w, s) in enumerate(zip(worse, significant)):
        for i, (w_inner, s_inner) in enumerate(zip(w, s)):
            significant[idx][i] = f"{w_inner.decode()} {s_inner.decode()}"

    return significant

create_dir_if_not_exist("tables")
pruning_methods_table = tabulate(generate_worse_significant_table(*compare_pruning_methods()), headers=pruning_values, showindex=datasets, tablefmt="latex_raw")
with open("tables/pruning_methods_table.txt", "w") as file:
    file.write(pruning_methods_table)
pruning_methods_table = tabulate(generate_worse_significant_table(*compare_pruning_methods()), headers=pruning_values, showindex=datasets, tablefmt="grid")
print("\nPruning random (vs pruning smallest values):\n", pruning_methods_table)

pruning_worst_f1_table = tabulate(generate_worse_significant_table(*compare_f1_scores_using_pruning_worst()), headers=pruning_values, showindex=datasets, tablefmt="latex_raw")
with open("tables/pruning_worst_f1_table.txt", "w") as file:
    file.write(pruning_worst_f1_table)
pruning_worst_f1_table = tabulate(generate_worse_significant_table(*compare_f1_scores_using_pruning_worst()), headers=pruning_values, showindex=datasets, tablefmt="grid")
print("\nPruning smallest values table (vs default model):\n", pruning_worst_f1_table)

pruning_random_f1_table = tabulate(generate_worse_significant_table(*compare_f1_scores_using_pruning_random()), headers=pruning_values, showindex=datasets, tablefmt="latex_raw")
with open("tables/pruning_random_f1_table.txt", "w") as file:
    file.write(pruning_random_f1_table)
pruning_random_f1_table = tabulate(generate_worse_significant_table(*compare_f1_scores_using_pruning_random()), headers=pruning_values, showindex=datasets, tablefmt="grid")
print("\nPruning random values table (vs default model):\n", pruning_random_f1_table)

exit()
for dataset_id, dataset in enumerate(datasets):
    print("\n", "-" * 20, dataset, "-" * 20)

    print("\n", "=" * 20, "ttest between pruning methods", "=" * 20)
    for pruning_id, pruning_val in enumerate(constants.PRUNING_VALUES):
        # first_score = pruned_scores[dataset_id, 0, i, :]
        # second_score = pruned_scores[dataset_id, 1, i, :]

        first_score = pruned_scores[dataset_id, :, 0, pruning_id]
        second_score = pruned_scores[dataset_id, :, 1, pruning_id]
        result = ttest_ind(first_score, second_score)

        first_score_mean = pruned_folds_mean[dataset_id, 0, pruning_id]
        second_score_mean = pruned_folds_mean[dataset_id, 1, pruning_id]

        print(f"Original mean score: {first_score_mean}")
        print(f"Pruning: {pruning_val * 100:.0f}% mean score: {second_score_mean}")
        print("Stat:", result.statistic)
        print("p-val:", f"{result.pvalue:.6f}")
        direction = "BETTER" if result.statistic < 0 else "WORSE"
        significance = "statistically significant" if result.pvalue < constants.P_VALUE else "NOT statistically SIGNIFICANT"
        print(f"{direction} result after pruning, {significance}")
        print()


    print("\n", "=" * 20, "ttest between pruning values f1 score and default score using prune_worst", "=" * 20)
    for pruning_id, pruning_val in enumerate(constants.PRUNING_VALUES):
        # first_score = pruned_scores[dataset_id, 0, 0, :]
        # second_score = pruned_scores[dataset_id, 0, i, :]

        first_score = pruned_scores[dataset_id, :, 0, 0]
        second_score = pruned_scores[dataset_id, :, 0, pruning_id]
        result = ttest_ind(first_score, second_score)

        first_score_mean = pruned_folds_mean[dataset_id, 0, 0]
        second_score_mean = pruned_folds_mean[dataset_id, 0, pruning_id]

        print(f"Original mean score: {first_score_mean}")
        print(f"Pruning: {pruning_val * 100:.0f}% mean score: {second_score_mean}")
        print("Stat:", result.statistic)
        print("p-val:", f"{result.pvalue:.6f}")
        direction = "BETTER" if second_score_mean > first_score_mean else "WORSE"
        significance = "statistically significant" if result.pvalue < constants.P_VALUE else "NOT statistically SIGNIFICANT"
        print(f"{direction} result after pruning, {significance}")
        print()

    print("\n", "=" * 20, "ttest between pruning values f1 score and default score using prune_random", "=" * 20)
    for pruning_id, pruning_val in enumerate(constants.PRUNING_VALUES):
        # first_score = pruned_scores[dataset_id, 1, 0, :]
        # second_score = pruned_scores[dataset_id, 1, i, :]

        first_score = pruned_scores[dataset_id, :, 1, 0]
        second_score = pruned_scores[dataset_id, :, 1, pruning_id]
        result = ttest_ind(first_score, second_score)

        first_score_mean = pruned_folds_mean[dataset_id, 1, 0]
        second_score_mean = pruned_folds_mean[dataset_id, 1, pruning_id]

        print(f"Original mean score: {first_score_mean}")
        print(f"Pruning: {pruning_val * 100:.0f}% mean score: {second_score_mean}")
        print("Stat:", result.statistic)
        print("p-val:", f"{result.pvalue:.6f}")
        direction = "BETTER" if second_score_mean > first_score_mean else "WORSE"
        significance = "statistically significant" if result.pvalue < constants.P_VALUE else "NOT statistically SIGNIFICANT"
        print(f"{direction} result after pruning, {significance}")
        print()
