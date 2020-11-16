from torch.utils.data import Dataset,DataLoader
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
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models , utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# torch.utils.data.Dataset

tic = time.time()

num_epochs = 3
num_classes =3
learning_rate = 0.001

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
            try:
                img_array=cv2.imread(os.path.join(images_dir,arr[0]))
                new_img_array=cv2.resize(img_array,(img_size,img_size))
                new_img_array = cv2.cvtColor(new_img_array, cv2.COLOR_BGR2RGB)
                human_data.append([new_img_array,arr[5]])
                data.append([new_img_array,arr[5]])
            except Exception as e:
                print("Data not included")
                print(arr[0])
create_data()
non_human_data=[]
def create_non_human_data():
       for i in range(len(non_human_csv)):
            arr=[]
            for j in non_human_csv.iloc[i]:
                   arr.append(j)
            try:
                img_array=cv2.imread(os.path.join(non_human_images_dir,arr[0]))
                new_img_array=cv2.resize(img_array,(img_size,img_size))
                new_img_array = cv2.cvtColor(new_img_array, cv2.COLOR_BGR2RGB)
                non_human_data.append([new_img_array,arr[1]])
                data.append([new_img_array,arr[1]])
            except Exception as e:
                print("Data not included")
                print(arr[0])

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
print("Training Data:")
print(len(X_train))
print("Testing Data:")
print(len(X_test))

import collections
print("0 =>")
print(lbl.inverse_transform([0]))
print("1 =>")
print(lbl.inverse_transform([1]))
print("2 =>")
print(lbl.inverse_transform([2]))

print("Training ")
print(collections.Counter(y_train))
print("Testing ")
print(collections.Counter(y_test))
# X_train = np.expand_dims(X_train, -1)
# X_test = np.expand_dims(X_test, -1)
X_train = np.array(X_train)
X_test = np.array(X_test)

class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(

            nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, padding=1),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=50, out_channels=75, kernel_size=3, padding=1),
            nn.BatchNorm2d(75),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(in_channels=75, out_channels=125, kernel_size=3, padding=1),
            nn.BatchNorm2d(125),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=2),
            nn.Dropout(p=0.25),
            nn.Linear(1, 3)
        )

    def forward(self, x):

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        return x

if __name__ == '__main__':
    freeze_support()
    model = CNN()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = MyDataset(X_train, y_train)
    train_loader = DataLoader(
    dataset,
    batch_size=10,
    shuffle=True,
    num_workers=2,
)
    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))

    dataset_test = MyDataset(X_test, y_test)
    test_loader = DataLoader(
        dataset_test
    )
    pred=[]
    leb=[]
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            pred.append(predicted)
            leb.append(labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

    toc = time.time()

    print('duration = ', toc - tic)
    leb = np.array(leb)
    pred = np.array(pred)
# # sparse_categorical_crossentropy: Computes the sparse categorical crossentropy loss. (For multiclass)
# model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
#
# # training the model for 10 epochs
# model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_test, y_test))
# test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
#
# x_test_final, y_test_final = dataset_test

print(classification_report(leb, pred,))
print("Confusion Matrix:\n", confusion_matrix(leb, pred))
