import datetime
import torch
import copy
from sklearn.model_selection import RepeatedStratifiedKFold
from torch.utils.data import DataLoader

from preprocessing import os_helper
from model import create_resnet_model
import numpy as np
import constants
import dataset_definition
import transformations
import os
from training import train_model
import glob


datasets = os.listdir(constants.DATASETS_DIR)

model_base = create_resnet_model(constants.NUM_CLASSES, fine_tuning=False)
rskf = RepeatedStratifiedKFold(n_repeats=constants.N_SPLITS, n_splits=constants.N_REPEATS, random_state=23)
scores = np.zeros((len(datasets), constants.N_SPLITS * constants.N_REPEATS))
os_helper.create_dir_if_not_exist(constants.MODELS_DIR)

current_time_dir = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os_helper.create_dir_if_not_exist(constants.MODELS_DIR)
os_helper.create_dir_if_not_exist(os.path.join(constants.MODELS_DIR, current_time_dir))

for dataset_id, dataset in enumerate(datasets):
    os_helper.create_dir_if_not_exist(os.path.join(constants.MODELS_DIR, current_time_dir, dataset))
    file_paths = glob.glob(os.path.join(constants.DATASETS_DIR, dataset, "**", "*.png"), recursive=True)
    paths_labels = {path: int(os.path.split(os.path.split(path)[0])[1]) for path in file_paths}

    X_file_paths = np.array(list(paths_labels.keys()))
    y_labels = np.array(list(paths_labels.values()))

    for fold_id, (train, test) in enumerate(rskf.split(X_file_paths, y_labels)):
        model = copy.deepcopy(model_base)
        optimizer = torch.optim.SGD(model.parameters(), lr=constants.LEARNING_RATE, momentum=constants.MOMENTUM)
        loss_func = torch.nn.CrossEntropyLoss()

        y_train = torch.from_numpy(y_labels[train]).type(torch.LongTensor)
        y_test = torch.from_numpy(y_labels[test]).type(torch.LongTensor)
        transformer = transformations.get_transform()

        dataset_train = dataset_definition.BreastDataset(paths=X_file_paths[train], labels=y_train, trans=transformer)
        dataset_test = dataset_definition.BreastDataset(paths=X_file_paths[test], labels=y_test, trans=transformer)

        dl_train = DataLoader(dataset=dataset_train, batch_size=constants.BATCH_SIZE, shuffle=True)
        dl_test = DataLoader(dataset=dataset_test, batch_size=constants.BATCH_SIZE, shuffle=False)
        loaders = {"train": dl_train, "test": dl_test}

        ################################################################################################################
        # model, acc_hist, best_acc = train_model(model, loaders, loss_func, optimizer, constants.NUM_EPOCHS)   # zmienić, żeby była lista preds zwracana


        # score = accuracy_score(y[test], y_pred)
        # scores[imputer_id, selector_id, fold_id, clf_id] = score

        # model, preds = train_model(model, loaders, loss_func, optimizer, constants.NUM_EPOCHS)   # zmienić, żeby była lista preds zwracana
        # pruning?

        ################################################################################################################
        scores[dataset_id, fold_id] = 0
        # scores[dataset_id, fold_id] = best_acc
        model_name = f"ai-model-pruning_fold-{fold_id}_score-{round(scores[dataset_id, fold_id])}.pth"
        model_save_path = os.path.join(constants.MODELS_DIR, current_time_dir, dataset, model_name)
        torch.save(model, model_save_path)


# Zapisanie wynikow
scores_name = "scores.npy"
scores_save_path = os.path.join(constants.MODELS_DIR, current_time_dir, scores_name)
np.save(scores_name, scores)
