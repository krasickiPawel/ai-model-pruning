from matplotlib import pyplot as plt
import os
import numpy as np
import constants
import matplotlib
matplotlib.use('TkAgg')


datasets = os.listdir(constants.DATASETS_DIR)
pruning_values = constants.PRUNING_VALUES
pruning_methods = ["prune_worst", "prune_random"]

scores_save_path_train = os.path.join(constants.MODELS_DIR, constants.TRAINED_MODELS_TO_PRUNE_CURRENT_DATETIME_DIR, constants.TRAINING_SCORES_NAME)
scores_save_path_prune = os.path.join(constants.PRUNING_DIR, constants.PRUNED_MODELS_TO_TEST_CURRENT_DATETIME_DIR, constants.PRUNING_SCORES_NAME)

trained_scores = np.load(scores_save_path_train)    # scores[dataset_id, fold_id] = f1_result
pruned_scores = np.load(scores_save_path_prune)     # scores[dataset_id, fold_id, pruning_method_id, pruning_value_id]


trained_scores_mean = np.mean(trained_scores, axis=1)
pruned_folds_mean = np.mean(pruned_scores, axis=1)

print(f"Score for pruning values {pruning_values}:",
      pruned_folds_mean)
# plt.savefig("fig1")
# plt.plot(trained_scores_mean)
# plt.savefig("fig1")
plt.rcParams["figure.figsize"] = [10.00, 7.50]
plt.rcParams["figure.autolayout"] = True
# fig, ax = plt.subplots(1, 2, figsize=(15, 7))
# # .imshow(img, cmap='magma')
# fig.savefig("fig1")

plt.xlabel("pruning values")
plt.title("Pruning by smallest values")
plt.ylabel("F1")
# plt.xticks(pruning_values)
print(pruned_folds_mean[0, 0, :])
plt.plot(pruning_values, list(pruned_folds_mean[0, 0, :]), marker='o', label=datasets[0])
plt.plot(pruning_values, list(pruned_folds_mean[1, 0, :]), marker='o', label=datasets[1])
plt.plot(pruning_values, list(pruned_folds_mean[2, 0, :]), marker='o', label=datasets[2])
plt.legend()
plt.show()



plt.xlabel("pruning values")
plt.title("Pruning by random values")
plt.ylabel("F1")
# plt.xticks(pruning_values)
print(pruned_folds_mean[0, 0, :])
plt.plot(pruning_values, list(pruned_folds_mean[0, 1, :]), marker='o', label=datasets[0])
plt.plot(pruning_values, list(pruned_folds_mean[1, 1, :]), marker='o', label=datasets[1])
plt.plot(pruning_values, list(pruned_folds_mean[2, 1, :]), marker='o', label=datasets[2])
plt.legend()
plt.show()
exit()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# z, x, y = pruned_folds_mean.nonzero()
# ax.scatter(x, y, z, c=z, alpha=1)
# plt.xlabel('Tablica 3 wymiarów zobrazowana na wykresie')
# plt.show()

################################################### mts2  ##############################################################




data = scores_folds_mean
z, x, y = data.nonzero()
ax.scatter(x, y, z, c=z, alpha=1)
plt.xlabel('Tablica 3 wymiarów zobrazowana na wykresie')
plt.show()


scores_imputers2d = np.mean(scores_folds_mean, axis=2)
scores_imputers = np.mean(scores_imputers2d, axis=1)
plt.bar(imputers, scores_imputers)
plt.xlabel('Porównanie metod radzenia sobie z brakującymi wartościami cech')
plt.show()

scores_selectors2d = np.mean(scores_folds_mean, axis=2)
scores_selectors = np.mean(scores_selectors2d, axis=1)
plt.bar(selectors, scores_selectors)
plt.xlabel('Porównanie metod selekcji cech')
plt.show()



scores_clfs2d = np.mean(scores_folds_mean, axis=2)
scores_clfs = np.mean(scores_clfs2d, axis=1)
plt.bar(clfs, scores_clfs)
plt.xlabel('Porównanie klasyfikatorów')
plt.show()





# fig, ax = plt.subplots(figsize=(9, 7))
# colors = ["blue", "green", "cyan", "yellow", "black"]
# print(len(assignments))
# for idx, assignment in enumerate(assignments):
#     for x in assignment:
#         ax.scatter(x[0], x[1], c=colors[idx])
#
# ax.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="+")
# fig.savefig("res.jpg")

