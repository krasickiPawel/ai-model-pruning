import torch
from copy import deepcopy
from sklearn.model_selection import RepeatedStratifiedKFold

from model import create_resnet_model
import numpy as np
import constants
import zip_reader


model_base = create_resnet_model()
rskf = RepeatedStratifiedKFold(n_repeats=constants.N_SPLITS, n_splits=constants.N_REPEATS, random_state=23)
scores = np.zeros((len(constants.DATASET_NAMES), constants.N_SPLITS * constants.N_REPEATS))


for dataset in constants.DATASET_NAMES:
    zip_file_path = f"{constants.DATASET_DIR}\\{dataset}\\{constants.ZIP_FILE_NAME}"
    zip_files = zip_reader.get_zip_files(zip_file_path)
    labels = zip_reader.get_labels(zip_files, *constants.DATASET_CANCER_LABELS.get(dataset))

    for fold_id, (train, test) in enumerate(rskf.split(zip_files, labels)):
        print(zip_files[train], labels[test])
        exit()

        model = deepcopy(model_base)
        # loader
        train(model) # def train(model, data_loaders_phases, loss_func, optimizer, num_epochs=1):
        # train_model()
        # test created model with predictions
        # calculate scores
        y_pred = clf.predict(X_selected[test])

        score = accuracy_score(y[test], y_pred)
        scores[imputer_id, selector_id, fold_id, clf_id] = score

# Zapisanie wynikow
file_date = datetime.datetime.now().time()
filename = f'results{file_date}'.replace(".", "_").replace(":", "_")
np.save(filename, scores)

#
# import numpy as np
#
# from sklearn.datasets import load_iris, load_breast_cancer
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.metrics import balanced_accuracy_score
# from sklearn.base import clone
#
# from sklearn.pipeline import Pipeline
#
# from random_clasifier import RandomClassifier
# from sklearn.neighbors import KNeighborsClassifier
#
# rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=100)
#
# DATSETS = [
#     load_iris(return_X_y=True),
#     load_breast_cancer(return_X_y=True)
# ]
#
# CLASSIFIERS = [
#     RandomClassifier(random_state=100),
#     KNeighborsClassifier(n_neighbors=1),
#     KNeighborsClassifier(n_neighbors=7),
# ]
#
# print(rskf.get_n_splits())
#
# scores = np.zeros(shape=(len(DATSETS), len(CLASSIFIERS), rskf.get_n_splits()))
#
# for ds_idx, (X, y) in enumerate(DATSETS):
#     # print('=' * 40)
#     for est_idx, est in enumerate(CLASSIFIERS):
#         # print('-' * 40)
#         for fold_idx, (train, test) in enumerate(rskf.split(X, y)):
#             clf = clone(est)
#             clf.fit(X[train], y[train])
#             y_pred = clf.predict(X[test])
#             scores[ds_idx, est_idx, fold_idx] = balanced_accuracy_score(y[test], y_pred)
#
# np.save("scores", scores)