from torch.utils.data import DataLoader
import glob
from sklearn.model_selection import train_test_split
import torch
from dataset import FingerTrainTestDataset, FingerValDataset
from transformations import train_test_trans, get_transform_test
import numpy as np
from labels import label_to_int


def load_train_test_dataset(dir_path, ts=.2, batch_size=8, input_required_size=224):
    trans_train, trans_test = train_test_trans(input_required_size)
    paths = glob.glob(f'{dir_path}/**/*.jpg', recursive=True)
    labels = []
    for path in paths:
        # label = path.split("\\")[-1].split("_")[0]
        label = path.split("\\")[-1][0]
        print(path)
        print(label)
        label = label_to_int.get(label)
        print(label, type(label))
        print()
        labels.append(label)
    # labels = [int(path[-5]) for path in paths]

    X_train, X_test, y_train, y_test = train_test_split(paths, labels, test_size=ts, random_state=42)

    y_train = torch.from_numpy(np.array(y_train)).type(torch.LongTensor)
    y_test = torch.from_numpy(np.array(y_test)).type(torch.LongTensor)

    dataset_train = FingerTrainTestDataset(paths=X_train, labels=y_train, trans=trans_train)
    dataset_test = FingerTrainTestDataset(paths=X_test, labels=y_test, trans=trans_test)

    dl_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

    data_loaders_phases = {
        "train": dl_train,
        "test": dl_test
    }

    return data_loaders_phases


def load_val_dir(dir_path, batch_size=16, input_required_size=224):
    trans_test = get_transform_test(input_required_size)
    paths = glob.glob(f'{dir_path}/*.png')
    paths += glob.glob(f'{dir_path}/*.jpg')
    paths += glob.glob(f'{dir_path}/*.jpeg')
    paths += glob.glob(f'{dir_path}/*.gif')
    if len(paths) > 1000:
        return None

    dataset_val = FingerValDataset(paths=paths, trans=trans_test)

    dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)

    return dataloader_val
