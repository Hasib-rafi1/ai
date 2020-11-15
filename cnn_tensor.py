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
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models , utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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




# building a linear stack of layers with the sequential model
model = models.Sequential()

# tf.keras.layers.Conv2D(
#     filters, kernel_size, strides=(1, 1), padding='valid', activation=None, input_shape=(32,32,3)
# )

model.add(layers.Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(50, 50, 3)))

# convolutional layer
model.add(layers.Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
# The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))

# flatten output of conv. Flattens the input. Does not affect the batch size.
model.add(layers.Flatten())

# hidden layer
# Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(250, activation='relu'))
model.add(layers.Dropout(0.3))
# output layer
model.add(layers.Dense(3, activation='softmax'))

# sparse_categorical_crossentropy: Computes the sparse categorical crossentropy loss. (For multiclass)
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 10 epochs
model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_test, y_test))
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
y_pred = model.predict(X_test)
print(classification_report(y_test, np.argmax(y_pred, axis=1)))
print("Confusion Matrix:\n", confusion_matrix(y_test, np.argmax(y_pred, axis=1)))
