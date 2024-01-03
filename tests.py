import os
import numpy as np
from tabulate import tabulate
from scipy.stats import ttest_ind

import constants


datasets = os.listdir(constants.DATASETS_DIR)
pruning_values = constants.PRUNING_VALUES

header_pruning_methods = ["pruning worst", "pruning random"]

# scores[dataset_id, fold_id] = f1_result
scores_save_path_train = os.path.join(constants.MODELS_DIR, constants.TRAINED_MODELS_TO_PRUNE_CURRENT_DATETIME_DIR, constants.TRAINING_SCORES_NAME)
# scores[dataset_id, fold_id, pruning_method_id, pruning_value_id]
scores_save_path_prune = os.path.join(constants.PRUNING_DIR, constants.PRUNED_MODELS_TO_TEST_CURRENT_DATETIME_DIR, constants.PRUNING_SCORES_NAME)

trained_scores = np.load(scores_save_path_train)
pruned_scores = np.load(scores_save_path_prune)

# mean po pruning methon - ale porównać te metody głównie osobno
pruned_folds_mean = np.mean(pruned_scores, axis=1)
table = tabulate(pruned_folds_mean[:, 0, :], tablefmt="grid", headers=pruning_values, showindex=datasets)
print(header_pruning_methods[0])
print(table)

table = tabulate(pruned_folds_mean[:, 1, :], tablefmt="grid", headers=pruning_values, showindex=datasets)
print(header_pruning_methods[1])
print(table)

table_avg = tabulate(zip(datasets, np.mean(trained_scores, axis=-1)), tablefmt="grid", headers=["dataset", "F1 score avg"])
print(table_avg)
print()
table_std = tabulate(zip(datasets, np.std(trained_scores, axis=-1)), tablefmt="grid", headers=["dataset", "F1 score std"])
print(table_std)
#
# table_p_avg = tabulate(zip(datasets, np.mean(pruned_scores, axis=1)), tablefmt="grid", headers=["dataset", "F1 score avg"])
# print(table_p_avg)
# print()
# table_p_std = tabulate(zip(datasets, np.std(pruned_scores, axis=1)), tablefmt="grid", headers=["dataset", "F1 score std"])
# print(table_p_std)



result = ttest_ind(pruned_folds_mean[:, 0, 0], pruned_folds_mean[:, 1, 0])
print("Stat:", result.statistic)
print("p-val:", f"{result.pvalue:.6f}")

for i, val in enumerate(constants.PRUNING_VALUES, 1):
    print(i)
    print(val)
    result = ttest_ind(pruned_folds_mean[:, 0, i], pruned_folds_mean[:, 1, i])
    print("Stat:", result.statistic)
    print("p-val:", f"{result.pvalue:.6f}")
    print()
