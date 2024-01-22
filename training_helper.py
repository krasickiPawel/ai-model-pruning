import copy

import torch
import constants
# from torchmetrics import F1Score
from sklearn.metrics import f1_score


def train_model(model, data_loaders, loss_func, optimizer, num_epochs=1):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    model = model.to(device)
    test_f1_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0
    torch.cuda.empty_cache()

    for epoch in range(num_epochs):
        print("-" * 40)
        print("Epoch {}/{}".format(epoch, constants.NUM_EPOCHS - 1))
        for phase in data_loaders:
            model.train() if phase == "train" else model.eval()
            # current_loss = 0.0
            all_preds = []
            all_labels = []

            for imgs, labels in data_loaders[phase]:
                imgs = imgs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(imgs)
                    loss = loss_func(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # current_loss += loss.item() * imgs.size(0)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu().data)

            torch_preds = torch.cat(all_preds)
            torch_labels = torch.cat(all_labels)

            f1_sklearn = f1_score(torch_labels, torch_preds)
            # f1_torchmetrics_obj = F1Score(task='binary')
            # f1_torchmetrics = f1_torchmetrics_obj(torch_preds, torch_labels)

            epoch_f1 = f1_sklearn

            print("{} F1: {:.4f}".format(phase, epoch_f1))
            if phase == "test" and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "test":
                test_f1_history.append(epoch_f1)

    for i, h in enumerate(test_f1_history):
        print(f"F1 history {i}:", h)
    print("Best val F1: {:4f}".format(best_f1))

    model.load_state_dict(best_model_wts)

    return model, best_f1
