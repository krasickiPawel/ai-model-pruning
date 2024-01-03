import copy
import glob
import os
import datetime
import numpy as np
import torch
from sklearn.model_selection import RepeatedStratifiedKFold
from torch.utils.data import DataLoader

import constants
import transformations
from dataset import BreastDataset
from preprocessing import os_helper
from pruning import prune_worst, prune_random
from predict import get_f1_score


datasets = os.listdir(constants.DATASETS_DIR)
rskf = RepeatedStratifiedKFold(n_repeats=constants.N_SPLITS, n_splits=constants.N_REPEATS, random_state=23)
pruning_methods = [prune_worst, prune_random]
pruning_values = [0.1, 0.2, 0.4, 0.6, 0.8]

unprunned_model_scores = 1
pruning_scores = len(pruning_values) + unprunned_model_scores
scores = np.zeros((len(datasets), constants.N_SPLITS * constants.N_REPEATS, len(pruning_methods), pruning_scores))

current_time_dir = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os_helper.create_dir_if_not_exist(constants.PRUNING_DIR)
os_helper.create_dir_if_not_exist(os.path.join(constants.PRUNING_DIR, current_time_dir))

processing_models_datetime_dir = "2024-01-03_01-05-25"

for dataset_id, dataset in enumerate(datasets):
    print()
    print("=" * 40)
    print(dataset)
    print()

    os_helper.create_dir_if_not_exist(os.path.join(constants.PRUNING_DIR, current_time_dir, dataset))
    model_paths = glob.glob(os.path.join(constants.MODELS_DIR, processing_models_datetime_dir, dataset, "*.pth"))
    print(model_paths)

    file_paths = glob.glob(os.path.join(constants.DATASETS_DIR, dataset, "**", "*.png"), recursive=True)
    paths_labels = {path: int(os.path.split(os.path.split(path)[0])[1]) for path in file_paths}

    X_file_paths = np.array(list(paths_labels.keys()))
    y_labels = np.array(list(paths_labels.values()))

    transformer = transformations.get_valid_transformer(X_file_paths[0])

    for fold_id, (train, test) in enumerate(rskf.split(X_file_paths, y_labels)):
        print()
        print("Fold {} ".format(fold_id) * 10)
        model_contains_fold_name = f"fold-{fold_id}"
        wanted_model_path = next((path for path in model_paths if model_contains_fold_name in path), None)
        print(wanted_model_path)

        model_base = torch.load(wanted_model_path)
        y_test = torch.from_numpy(y_labels[test]).type(torch.LongTensor)
        dataset_test = BreastDataset(paths=X_file_paths[test], labels=y_test, trans=transformer)
        dl_test = DataLoader(dataset=dataset_test, batch_size=constants.BATCH_SIZE, shuffle=False)

        for pruning_method_id, pruning_method in enumerate(pruning_methods):
            scores[dataset_id, fold_id, pruning_method_id, 0] = get_f1_score(model_base, dl_test)

            for pruning_value_id, pruning_value in enumerate(pruning_values, 1):
                model = copy.deepcopy(model_base)
                pruning_method(model, pruning_value)
                scores[dataset_id, fold_id, pruning_method_id, pruning_value_id] = get_f1_score(model, dl_test)


scores_name = "pruning_scores.npy"
scores_save_path = os.path.join(constants.PRUNING_DIR, current_time_dir, scores_name)
np.save(scores_save_path, scores)



