from argparse import ArgumentParser
import os.path
import time

import cv2
import numpy as np

import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms

from lines_segmentation.binarization import binarize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Binarization(object):
    def __init__(self):
        pass

    def __call__(self, img: Image) -> Image:
        """
        :param img: (PIL): Image
        :return: binarized image (PIL)
        """
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = binarize(img)
        return Image.fromarray(img)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


def train(dataset_path: str, save_model_path: str) -> None:

    transforms_list = transforms.Compose([
        Binarization(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dir_name = "train"
    val_dir_name = "val"

    train_dataset = datasets.ImageFolder(os.path.join(dataset_path, train_dir_name), transforms_list)
    val_dataset = datasets.ImageFolder(os.path.join(dataset_path, val_dir_name), transforms_list)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=12, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=12, shuffle=False)

    print('Train dataset size:', len(train_dataset))
    print('Val dataset size:', len(val_dataset))
    class_names = train_dataset.classes
    print('Class names:', class_names)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 30
    start_time = time.time()
    for epoch in range(num_epochs):
        print("Epoch {} running".format(epoch))  # (printing message)

        """ Training Phase """
        model.train()
        running_loss = 0.
        running_corrects = 0
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects / len(train_dataset) * 100.
        print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc,
                                                                           time.time() - start_time))
        torch.save(model.state_dict(), save_model_path)

        """ Testing Phase """
        model.eval()
        with torch.no_grad():
            running_loss = 0.
            running_corrects = 0
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(val_dataset)
            epoch_acc = running_corrects / len(val_dataset) * 100.
            print('[Test #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc,
                                                                              time.time() - start_time))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the train and val datasets')
    parser.add_argument('--save_model_path', type=str, required=True, help='Where to save trained model')
