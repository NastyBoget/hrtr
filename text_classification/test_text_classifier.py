import os.path
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(dataset_path: str, save_model_path: str) -> None:
    dataset_name = "text_classification_split"
    dataset_path = os.path.join(dataset_path, dataset_name)
    test_dir_name = "test"
    transforms_list = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.ImageFolder(os.path.join(dataset_path, test_dir_name), transforms_list)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=192, shuffle=True)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(os.path.join(save_model_path, "text_classifier_resnet18.pth")))
    model.to(device)

    model.eval()
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        running_loss = 0.
        running_corrects = 0
        for i, (inputs, labels) in enumerate(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(test_dataset)
        epoch_acc = running_corrects / len(test_dataset) * 100.
        print('[Test] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.
              format(epoch_loss, epoch_acc, time.time() - start_time))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the directory with the test dataset, '
                                                                     'the name of dataset is text_classification_split')
    parser.add_argument('--save_model_path', type=str, required=True, help='Directory with saved trained model named '
                                                                           'text_classifier_resnet18.pth')
    args = parser.parse_args()
    test(args.data_path, args.save_model_path)
