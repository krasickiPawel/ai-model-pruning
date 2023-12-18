import datetime

import torch
import copy
from sklearn.model_selection import RepeatedStratifiedKFold
from torch.utils.data import DataLoader

from model import create_resnet_model
import numpy as np
import constants
import zip_reader
import dataset_definition
import transformations
import time
import os
from training import train
from sklearn.metrics import accuracy_score


model_base = create_resnet_model(constants.NUM_CLASSES, pretrained=True, fine_tuning=False)
rskf = RepeatedStratifiedKFold(n_repeats=constants.N_SPLITS, n_splits=constants.N_REPEATS, random_state=23)
scores = np.zeros((len(constants.DATASET_NAMES), constants.N_SPLITS * constants.N_REPEATS))


for dataset_id, dataset in enumerate(constants.DATASET_NAMES):
    zip_file_path = os.path.join(constants.DATASET_DIR, dataset, constants.ZIP_FILE_NAME)
    zip_files = np.array(zip_reader.get_zip_files(zip_file_path))
    labels = np.array(zip_reader.get_labels(zip_files, *constants.DATASET_CANCER_LABELS.get(dataset)))

    for fold_id, (train, test) in enumerate(rskf.split(zip_files, labels)):
        model = copy.deepcopy(model_base)
        optimizer = torch.optim.SGD(model.parameters(), lr=constants.LEARNING_RATE, momentum=constants.MOMENTUM)
        loss_func = torch.nn.CrossEntropyLoss()

        y_train = torch.from_numpy(labels[train]).type(torch.LongTensor)
        y_test = torch.from_numpy(labels[test]).type(torch.LongTensor)
        transformer = transformations.get_transform_train()

        dataset_train = dataset_definition.BreastDataset(paths=zip_files[train], labels=y_train, trans=transformer)
        dataset_test = dataset_definition.BreastDataset(paths=zip_files[test], labels=y_test, trans=transformer)

        dl_train = DataLoader(dataset=dataset_train, batch_size=constants.BATCH_SIZE, shuffle=True)
        dl_test = DataLoader(dataset=dataset_test, batch_size=constants.BATCH_SIZE, shuffle=False)
        loaders = {"train": dl_train, "test": dl_test}

        ################################################################################################################
        model, acc_hist, best_acc = train(
            model,
            loaders,
            loss_func,
            optimizer,
            constants.NUM_EPOCHS
        )

        # pruning?

        print("Finetuning enabled")
        for param in model.parameters():
            param.requires_grad = True

        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=constants.MOMENTUM)
        model, acc_hist, best_acc = train(
            model,
            loaders,
            loss_func,
            optimizer,
            constants.NUM_EPOCHS
        )

        scores[dataset_id, fold_id] = best_acc
        torch.save(model, f"models/breast_model_pruning_{dataset}_{fold_id}_{best_acc:.4f}.pth")
        ################################################################################################################
        # score = accuracy_score(y[test], y_pred)
        # scores[imputer_id, selector_id, fold_id, clf_id] = score

# Zapisanie wynikow
file_date = datetime.datetime.now().time()
filename = f'results_{file_date}'.replace(".", "_").replace(":", "_")
np.save(filename, scores)


