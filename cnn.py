from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from multiprocessing import Process, freeze_support
import time
import os
import cv2
import csv
import random

tic = time.time()

# num_epochs = 4
# num_classes = 10
# learning_rate = 0.001

print("Data preprocessing start....")
images_dir = os.path.join('import_data/data/images')
non_human_images_dir = os.path.join('import_data/data/non_human')
train_csv= pd.read_csv(os.path.join("import_data/train.csv"))
non_human_csv= pd.read_csv(os.path.join("import_data/non_human.csv"))
# print(len(train_csv))
# print(len(os.listdir(images_dir)))
# print(len(non_human_csv))

options=['face_with_mask','face_without_mask']
train= train_csv[train_csv['classname'].isin(options)]
train.sort_values('name',axis=0,inplace=True)


img_size=50
data=[]
human_data=[]
def create_data():
       for i in range(len(train)):
            arr=[]
            for j in train.iloc[i]:
                   arr.append(j)
            img_array=cv2.imread(os.path.join(images_dir,arr[0]),cv2.IMREAD_GRAYSCALE)
            crop_image = img_array[arr[2]:arr[4],arr[1]:arr[3]]
            new_img_array=cv2.resize(crop_image,(img_size,img_size))
            human_data.append([new_img_array,arr[5]])
            data.append([new_img_array,arr[5]])
create_data()
non_human_data=[]
def create_non_human_data():
       for i in range(len(non_human_csv)):
            arr=[]
            for j in non_human_csv.iloc[i]:
                   arr.append(j)
            try:
                if arr[0] =="name":
                    img_array=cv2.imread(os.path.join(non_human_images_dir,arr[0]),cv2.IMREAD_GRAYSCALE)
                    new_img_array=cv2.resize(img_array,(img_size,img_size))
                    non_human_data.append([new_img_array,arr[1]])
                    data.append([new_img_array,arr[1]])
            except Exception as e:
                print(arr[0])
                print(str(e))

create_non_human_data()
# print(len(human_data))
# print(len(non_human_data))

final_data = random.sample(data,len(data))

x=[]
y=[]
for features, labels in final_data:
    x.append(features)
    y.append(labels)

lbl=LabelEncoder()
y=lbl.fit_transform(y)

print("Data preprocessing end....")

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(len(X_train))
print(len(y_train))

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
