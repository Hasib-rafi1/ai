from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets
from multiprocessing import Process, freeze_support
import time
import pandas as pd
import os
import cv2
import random
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

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

            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(832, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512,3)
        )

    def forward(self, x):

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x

if __name__ == '__main__':
    freeze_support()
    print("Data preprocessing start....")
    images_dir = os.path.join('import_data/data/images')
    non_human_images_dir = os.path.join('import_data/data/non_human')
    PATH = os.path.join('torch_k.pt')
    train_csv= pd.read_csv(os.path.join("import_data/train.csv"))
    non_human_csv= pd.read_csv(os.path.join("import_data/non_human.csv"))
    # print(len(train_csv))

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
                    crop_image = img_array[arr[2]:arr[4],arr[1]:arr[3]]
                    new_img_array=cv2.resize(crop_image,(img_size,img_size))
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

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)


    tic = time.time()

    num_epochs = 10
    num_classes =3
    learning_rate = 0.001

    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset= MyDataset(data= X_test, target= y_test, transform=transform)

    test_loader_final = torch.utils.data.DataLoader(testset, batch_size=10000,
                                              shuffle=False, num_workers=2)
    model = None
    k_fold_count = 0
    kfold_result = []
    def train_model(x_trn, x_val, y_trn, y_val, num_epochs, best_lwlrap, weight_file_name, transform,batch_size,model,criterion,optimizer):
        global k_fold_count
        global kfold_result
        k_fold_count = k_fold_count+1
        print("KFOLD number: "+str(k_fold_count))
        trainset = MyDataset(data= X_train, target= y_train, transform=transform)


        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   shuffle=True, num_workers=2)

        testset= MyDataset(data= X_test, target= y_test, transform=transform)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  shuffle=False, num_workers=2)



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

        model.eval()

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        lwlrap = ((correct / total) * 100)
        a = {'kfold': k_fold_count, 'score': lwlrap}
        with open('listfile.txt', 'a') as filehandle:
                filehandle.write('%s\n' % a)

        if lwlrap > best_lwlrap:
            best_lwlrap = lwlrap
            if os.path.exists(weight_file_name):
                os.remove(weight_file_name)
            torch.save(model,weight_file_name)


        return {
            'best_lwlrap': best_lwlrap,
            'weight_file_name' : weight_file_name,
            'model': model
        }


    if os.path.exists(PATH):
        model = torch.load(PATH)
        model.eval()
        # open file and read the content in a list
        with open('listfile.txt', 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                currentPlace = line[:-1]

                # add item to the list
                kfold_result.append(currentPlace)
    else:
        best_lwlrap = 0
        weight_file_name = PATH
        model = CNN()

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        kf = KFold(n_splits=10, random_state=23, shuffle=True)
        for train_index, test_index in kf.split(np.arange(len(X_train))):
            X_train_k, X_test_k = X_train[train_index], X_train[test_index]
            y_train_k, y_test_k = y_train[train_index], y_train[test_index]
            k_fold_return = train_model(X_train_k, X_test_k, y_train_k, y_test_k, num_epochs, best_lwlrap, weight_file_name, transform, 32,model,criterion,optimizer)
            best_lwlrap = k_fold_return["best_lwlrap"]
            model = k_fold_return["model"]
        model = torch.load(PATH)

    all_pred_final=torch.tensor([])
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader_final:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_pred_final = torch.cat((all_pred_final,outputs),dim=0)
        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

    toc = time.time()

    print('duration = ', toc - tic)
    print(classification_report(testset.target.numpy(),np.argmax(all_pred_final.numpy(), axis=1)))
    print("Confusion Matrix:\n", confusion_matrix(testset.target.numpy(),np.argmax(all_pred_final.numpy(), axis=1)))
    print(kfold_result)
