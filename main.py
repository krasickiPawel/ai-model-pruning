from init_model import create_resnet_model
import torch
from loaders import load_train_test_dataset
from train import train


def main():
    # train_test_dir = "train_test_balanced"
    train_test_dir = r"C:\Users\Paweł\Documents\_studia\_mgr\1 sem\zastosowania informatyki w gospodarce\projekt\zdjęcia rąk"
    num_epochs = 10
    fine_tuning = True
    batch_size = 8
    learning_rate = 0.01
    pretrained = True
    test_size = 0.2
    # input_required_size = 224
    input_required_size = 480

    # num_classes = 2
    num_classes = 5

    model = create_resnet_model(num_classes, pretrained, fine_tuning=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    dls = load_train_test_dataset(
        train_test_dir,
        ts=test_size,
        batch_size=batch_size,
        input_required_size=input_required_size,
    )
    loss_func = torch.nn.CrossEntropyLoss()

    model, acc_hist, best_acc = train(model, dls, loss_func, optimizer, 10)

    print("Finetuning enabled")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model, acc_hist, best_acc = train(model, dls, loss_func, optimizer, 20)

    # MODEL_PATH = "fun_ai_model_finger_heart.pth"
    MODEL_PATH = "ziwg.pth"
    torch.save(model, MODEL_PATH)
    # torch.save(model.state_dict(), "fun_ai_model_finger_heart_wts.pth")


if __name__ == "__main__":
    main()
