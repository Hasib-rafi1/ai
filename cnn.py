from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from multiprocessing import Process, freeze_support
import time
import os
tic = time.time()

num_epochs = 4
num_classes = 10
learning_rate = 0.001

images_dir = os.path.join('import_data/data/images')
train_csv= pd.read_csv(os.path.join("import_data/train.csv"))
print(len(train_csv))
print(len(os.listdir(images_dir)))
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,
#                                            shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=1000,
#                                           shuffle=False, num_workers=2)
#
# classes = ('face_with_mask', 'face_without_mask', 'Not a Person')
#
# class CNN(nn.Module):
#
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv_layer = nn.Sequential(
#
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#
#         self.fc_layer = nn.Sequential(
#             nn.Dropout(p=0.1),
#             nn.Linear(8 * 8 * 64, 1000),
#             nn.ReLU(inplace=True),
#             nn.Linear(1000, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.1),
#             nn.Linear(512, 10)
#         )
#
#     def forward(self, x):
#
#         # conv layers
#         x = self.conv_layer(x)
#
#         # flatten
#         x = x.view(x.size(0), -1)
#
#         # fc layer
#         x = self.fc_layer(x)
#
#         return x
#
# if __name__ == '__main__':
#     freeze_support()
#     model = CNN()
#
#     # Loss and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#     # Train the model
#     total_step = len(train_loader)
#     loss_list = []
#     acc_list = []
#     for epoch in range(num_epochs):
#         for i, (images, labels) in enumerate(train_loader):
#             # Forward pass
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss_list.append(loss.item())
#
#             # Backprop and perform Adam optimisation
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # Track the accuracy
#             total = labels.size(0)
#             _, predicted = torch.max(outputs.data, 1)
#             correct = (predicted == labels).sum().item()
#             acc_list.append(correct / total)
#
#             if (i + 1) % 100 == 0:
#                 print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
#                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
#                               (correct / total) * 100))
#
#     model.eval()
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for images, labels in test_loader:
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#         print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))
#
#     toc = time.time()
#
#     print('duration = ', toc - tic)
