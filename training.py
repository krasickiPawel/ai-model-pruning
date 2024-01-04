import datetime
import torch
import copy
from sklearn.model_selection import RepeatedStratifiedKFold
from torch.utils.data import DataLoader

from preprocessing import os_helper
from model import create_resnet_model
import numpy as np
import constants
from dataset import BreastDataset
import transformations
import os
from training_helper import train_model
import glob


datasets = os.listdir(constants.DATASETS_DIR)

model_base = create_resnet_model(constants.NUM_CLASSES, fine_tuning=False)
for param in model_base.parameters():
    param.requires_grad = True

rskf = RepeatedStratifiedKFold(n_repeats=constants.N_SPLITS, n_splits=constants.N_REPEATS, random_state=23)
scores = np.zeros((len(datasets), constants.N_SPLITS * constants.N_REPEATS))

current_time_dir = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os_helper.create_dir_if_not_exist(constants.MODELS_DIR)
os_helper.create_dir_if_not_exist(os.path.join(constants.MODELS_DIR, current_time_dir))


for dataset_id, dataset in enumerate(datasets):
    print()
    print("=" * 40)
    print(dataset)
    print()

    os_helper.create_dir_if_not_exist(os.path.join(constants.MODELS_DIR, current_time_dir, dataset))
    file_paths = glob.glob(os.path.join(constants.DATASETS_DIR, dataset, "**", "*.png"), recursive=True)
    paths_labels = {path: int(os.path.split(os.path.split(path)[0])[1]) for path in file_paths}

    X_file_paths = np.array(list(paths_labels.keys()))
    y_labels = np.array(list(paths_labels.values()))

    transformer = transformations.get_valid_transformer(X_file_paths[0])

    for fold_id, (train, test) in enumerate(rskf.split(X_file_paths, y_labels)):
        print()
        print("Fold {} ".format(fold_id) * 10)
        model = copy.deepcopy(model_base)
        optimizer = torch.optim.SGD(model.parameters(), lr=constants.LEARNING_RATE, momentum=constants.MOMENTUM)
        loss_func = torch.nn.CrossEntropyLoss()

        y_train = torch.from_numpy(y_labels[train]).type(torch.LongTensor)
        y_test = torch.from_numpy(y_labels[test]).type(torch.LongTensor)

        dataset_train = BreastDataset(paths=X_file_paths[train], labels=y_train, trans=transformer)
        dataset_test = BreastDataset(paths=X_file_paths[test], labels=y_test, trans=transformer)

        dl_train = DataLoader(dataset=dataset_train, batch_size=constants.BATCH_SIZE, shuffle=True)
        dl_test = DataLoader(dataset=dataset_test, batch_size=constants.BATCH_SIZE, shuffle=False)
        loaders = {"train": dl_train, "test": dl_test}

        model, f1_result = train_model(model, loaders, loss_func, optimizer, constants.NUM_EPOCHS)
        scores[dataset_id, fold_id] = f1_result

        model_name = f"ai-model-pruning_fold-{fold_id}_score-{round(scores[dataset_id, fold_id] * 100)}.pth"
        model_save_path = os.path.join(constants.MODELS_DIR, current_time_dir, dataset, model_name)
        torch.save(model, model_save_path)


scores_save_path = os.path.join(constants.MODELS_DIR, current_time_dir, constants.TRAINING_SCORES_NAME)
np.save(scores_save_path, scores)
