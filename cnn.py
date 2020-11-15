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
                img_array=cv2.imread(os.path.join(images_dir,arr[0]),cv2.IMREAD_GRAYSCALE)
                crop_image = img_array[arr[2]:arr[4],arr[1]:arr[3]]
                new_img_array=cv2.resize(crop_image,(img_size,img_size))
                human_data.append([new_img_array,arr[5]])
                data.append([new_img_array,arr[5]])
            except Exception as e:
                print(arr[0])
                print(str(e))
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
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
X_train = np.array(X_train)
X_test = np.array(X_test)
# X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
# X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

# one-hot encoding using keras' numpy-related utilities
n_classes = 3
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = utils.to_categorical(y_train, n_classes)
Y_test = utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)


# building a linear stack of layers with the sequential model
model = models.Sequential()
model.add(layers.Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(50, 50, 1)))

# convolutional layer
model.add(layers.Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.25))

# flatten output of conv
model.add(layers.Flatten())

# hidden layer
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(250, activation='relu'))
model.add(layers.Dropout(0.3))
# output layer
model.add(layers.Dense(3, activation='softmax'))

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#
# print(X_train.shape)
# print(y_train.shape)
# model.compile(optimizer='sgd',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#               metrics=['accuracy'])
#
# history = model.fit(X_train, y_train, epochs=10)
#
# test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 10 epochs
model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))
test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)
print(test_acc)
