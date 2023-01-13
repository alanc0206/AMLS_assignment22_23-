#Import
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image
import pandas as pd
import numpy as np
from natsort import natsorted
from sklearn.metrics import accuracy_score
import random

#Use GPU if avaliable, if not, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyperparameters
type_of_label = 'face_shape'
num_class = 5
lr = 0.001
batch_size = 64
epochs = 50

#Transform
transform = transforms.Compose([transforms.CenterCrop((300,300)), #Crop the image because 500*500 is too big and most pixels are not useful
                                transforms.Resize((64, 64)), #Resized the image to 64*64 for faster processing
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#CNN class which defines the model
class CNN (nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.drop_out = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(8192, 2048)
        self.fc2 = nn.Linear(2048, num_class)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

#Early stopper class for preventing overfitting
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.00001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

#Dataset class for storing data and its corresponding label
class Datasets (Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.label[index]
        return image, torch.tensor(label)

#Initiate empty arrays for storing data and labels
train_data = []
train_label = []
val_data = []
val_label = []
test_data = []
test_label = []

#Load train dataset
train_data_dir = '../Datasets/dataset_AMLS_22-23/cartoon_set/img/'
train_label_dir = '../Datasets/dataset_AMLS_22-23/cartoon_set/labels.csv'

train_label_data = pd.read_csv(train_label_dir, sep='\t')
for label in train_label_data[type_of_label]:
    label = 0 if label == -1 else label
    train_label.append(label)

data = os.listdir(train_data_dir)
data = natsorted(data)

for f in data:
    path = os.path.join(train_data_dir, f)
    img = transform(Image.open(path).convert("RGB"))
    train_data.append(img)

#Allocate part of the train dataset as the validation set
ratio = 0.2
for i in range(int(ratio* len(train_data))):
     r = random.randrange(len(train_data))
     val_data.append(train_data.pop(r))
     val_label.append(train_label.pop(r))

#Load test dataset
test_data_dir = '../Datasets/dataset_AMLS_22-23_test/cartoon_set_test/img'
test_label_dir = '../Datasets/dataset_AMLS_22-23_test/cartoon_set_test/labels.csv'

test_label_data = pd.read_csv(test_label_dir, sep='\t')
for label in test_label_data[type_of_label]:
    label = 0 if label == -1 else label
    test_label.append(label)

data = os.listdir(test_data_dir)
data = natsorted(data)

for f in data:
    path = os.path.join(test_data_dir, f)
    img = transform(Image.open(path).convert("RGB"))
    test_data.append(img)

#Store data and its corresponding label for each set
train_dataset = Datasets(train_data, train_label)
val_dataset = Datasets(val_data, val_label)
test_dataset = Datasets(test_data, test_label)

#DataLoader
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, earlystopper, loss function and optimizer
model = CNN().to(device)
early_stopper = EarlyStopper(patience=3)

loss_func = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

#Train model
print('Start training')
for epoch in range (epochs):
    model.train() #Start training
    train_loss_list = []
    train_truth = []
    train_pred = []
    for step, (data, label) in enumerate(train_dataloader):
        data = data.to(device)
        label = label.to(device)
        train_truth.extend(label.tolist()) #Store the truth
        output = model(data) #Input the data into the model
        loss = loss_func(output, label) #Calculate loss
        train_loss_list.append(loss.item()) #Store the lass
        pred = torch.argmax(output, dim=1) #Predict the class of the output
        train_pred.extend(pred.tolist()) #Store the prediction

        optimizer.zero_grad() #Reset gradient to 0
        loss.backward() #Back propogate
        optimizer.step() #Update the calculated value

    train_loss = np.mean(train_loss_list) #Calculate mean loss of each epoch
    train_accuracy = accuracy_score(train_truth, train_pred) #Calculate accuracy of each epoch

    model.eval() #Start evaluating
    val_truth = []
    val_pred = []
    val_loss_list = []
    with torch.no_grad():
        for step, (data, label) in enumerate(val_dataloader):
            data = data.to(device)
            label = label.to(device)
            val_truth.extend(label.tolist())
            output = model(data)
            loss = loss_func(output, label)
            val_loss_list.append(loss.item())
            pred = torch.argmax(output, dim=1)
            val_pred.extend(pred.tolist())

    val_loss = np.mean(val_loss_list)
    val_accuracy = accuracy_score(val_truth, val_pred)
    print(f'Epoch: {epoch+1} | Training loss: {train_loss:.4f} | Training accuracy: {train_accuracy:.4f} |'
          f' Validation loss: {val_loss:.4f} | Validation accuracy: {val_accuracy:.4f}') #Print training and validation loss and accuracy

    if early_stopper.early_stop(val_loss): #Stop when validation loss is not improving enough
        torch.save(model.state_dict(), './model.pth') #Store the model
        print(f'The model is not improving because validation loss is not decreasing enough.'
              f'Training stopped after {epoch+1} epoch')
        break

#Testing the model
trained_model_dir = './model.pth'
isFile = os.path.isfile(trained_model_dir)
#Load and test the model if it exists
if isFile:
    test_model = CNN().to(device)
    test_model.load_state_dict(torch.load(trained_model_dir)) #Load model
    test_model.eval()
    test_truth = []
    test_pred = []
    test_loss_list = []
    with torch.no_grad():
        for step, (data, label) in enumerate(test_dataloader):
            data = data.to(device)
            label = label.to(device)
            test_truth.extend(label.tolist())
            output = test_model(data)
            loss = loss_func(output, label)
            test_loss_list.append(loss.item())
            pred = torch.argmax(output, dim=1)
            test_pred.extend(pred.tolist())

    test_loss = np.mean(test_loss_list)
    test_accuracy = accuracy_score(test_truth, test_pred)
    print(f' Testing loss: {test_loss:.4f} | Testing accuracy: {test_accuracy:.4f}') #Print testing loss and accuracy