from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score
from sklearn.base import clone
from training import train
from model import create_resnet_model
import datetime
import numpy as np
from zip_reader import get_preprocessed_dataset_and_labels


n_splits = 5
n_repeats = 2
# df_best_features = None

# models = {
#     "ResNet18": "resnet18",
#     "TenDrugi": ""
# }
model = create_resnet_model()

dataset_names = [
    "BreaKHis400X",
    "BreastUltrasoundImagesDataset(BUSI)",
    "UltrasoundBreastImagesforBreastCancer",
    # "BreastHistopathologyImages"
]
# train_files_no_cancer, train_files_cancer, test_files_no_cancer, test_files_cancer

rskf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=23)

scores = np.zeros((len(dataset_names), n_splits * n_repeats))


for dataset in dataset_names:
    paths_with_labels = get_preprocessed_dataset_and_labels(dataset)
    X = [element[0] for element in paths_with_labels]
    y = [element[1] for element in paths_with_labels]

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
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


