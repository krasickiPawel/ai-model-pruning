import torch
from sklearn.metrics import f1_score


def get_f1_score(model, data_loader):
    all_preds = []
    all_labels = []
    for imgs, labels in data_loader:
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        all_preds.append(preds)
        all_labels.append(labels)

    torch_preds = torch.cat(all_preds)
    torch_labels = torch.cat(all_labels)

    return f1_score(torch_labels, torch_preds)
