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
