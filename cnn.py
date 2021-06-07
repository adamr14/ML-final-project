#Convolutional Neural Network written for Machine Learning Project, Spring 2021
#Author: Carter Poythress
#Teammates: Austin Fuller and Adam Rankin
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys

batch_size = 64
epochs = 20
learning_rate = 0.001

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=3, padding=1)

        self.pool3 = nn.AvgPool2d(kernel_size=7)

        self.drop = nn.Dropout2d(0.1)

    def forward(self, x):
        x = self.drop(F.relu(self.conv1(x)))
        x = self.drop(F.relu(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop(F.relu(self.conv3(x)))
        x = self.drop(F.relu(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop(F.relu(self.conv5(x)))
        x = self.conv6(x)
        x = self.pool3(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

def load_data(filename, cuda):
    #read csv into a dataframe and normalize to range [0,1]
    train_df = pd.read_csv(filename, dtype=np.float32)
    train_features = train_df.loc[:, train_df.columns != 'label'].to_numpy() / 255
    train_labels = train_df["label"].to_numpy()

    if(cuda):#sets the features and labels to correct datatype, depending on if cuda cores are available
        features = torch.from_numpy(train_features.reshape(-1, 1, 28, 28)).type(torch.cuda.FloatTensor)
        labels = torch.from_numpy(train_labels.reshape(-1)).type(torch.cuda.LongTensor)
    else:
        features = torch.from_numpy(train_features.reshape(-1, 1, 28, 28)).type(torch.FloatTensor)
        labels = torch.from_numpy(train_labels.reshape(-1)).type(torch.LongTensor)

    return torch.utils.data.TensorDataset(features, labels)

def train_model(model, optimizer, train_dataloader, device):
    #iterates over the training data, using neg. log likelihood to calculate error,
    #and back-propagates the error for training
    model.train()
    for feature, label in train_dataloader:
        feature, label = feature.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(feature)
        computed_loss = F.nll_loss(output, label)
        computed_loss.backward()
        optimizer.step()


def test_model(model, test_dataloader, device, epoch_num):
    model.eval()
    computed_loss = 0
    correct = 0
    with torch.no_grad():
        for feature, label in test_dataloader:
            feature, label = feature.to(device), label.to(device)
            output = model(feature)
            computed_loss += F.nll_loss(output, label, reduction="sum").item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    computed_loss = computed_loss / len(test_dataloader.dataset)
    computed_accuracy = correct / len(test_dataloader.dataset)
    print("-----------------------------------------------------")
    print("Epoch #:", epoch_num)
    print("Average loss on the test set:", computed_loss)
    print("Overall accuracy on the test set:", computed_accuracy)
    print("-----------------------------------------------------")

    #save the model if it has the highest accuracy
    if computed_accuracy >= 0.99325: #this accuracy score is manually adjusted as higher accuracy is achieved through testing
        print("New Highest Accuracy Achieved! Saving this model...")
        torch.save(model.state_dict(), "./best_cnn_saved_model")

    return computed_loss, computed_accuracy

def main():
    #checking for cuda cores to see if we can calculate on GPU, this is MUCH faster than CPU
    #if you have an Nvidia GPU and want to run on CPU for some reason, uncomment the line below
    cuda = torch.cuda.is_available()
    #cuda = False
    train_dataset = load_data("./dataset/train.csv", cuda)
    test_dataset = load_data("./dataset/test.csv", cuda)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if cuda else "cpu")
    model = Cnn().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []
    acc_history = []
    for epoch in range(epochs):
        train_model(model, optimizer, train_dataloader, device)
        loss, acc = test_model(model, test_dataloader, device, (epoch+1))

        loss *= 100
        acc *= 100
        loss_history.append(loss)
        acc_history.append(acc)


    ind = np.arange(epochs)
    wid = 0.25

    plt.bar(ind, loss_history, label="Model Loss", width=wid,  color = "r")
    plt.bar(ind+wid, acc_history, label="Model Accuracy", width=wid,  color = "b")
    plt.xticks(range(epochs), range(epochs))
    plt.ylabel("Percentage")
    plt.xlabel("Epochs")
    plt.title("Accuracy and Loss Per Epoch")
    plt.legend()
    plt.show()

def test_best_model():
    #loads and runs the saved CNN model testing it against the test dataset
    cuda = torch.cuda.is_available()

    test_dataset = load_data("./dataset/test.csv", cuda)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if cuda else "cpu")
    model = Cnn()

    if cuda:
        model.load_state_dict(torch.load("./best_cnn_saved_model"))
    else:
        model.load_state_dict(torch.load("./best_cnn_saved_model", map_location=device))

    model.to(device)

    model.eval()
    computed_loss = 0
    correct = 0
    with torch.no_grad():
        for feature, label in test_dataloader:
            feature, label = feature.to(device), label.to(device)
            output = model(feature)
            computed_loss += F.nll_loss(output, label, reduction="sum").item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    computed_loss = computed_loss / len(test_dataloader.dataset)
    computed_accuracy = correct / len(test_dataloader.dataset)
    print("-----------------------------------------------------")
    print("Average loss on the test set:", computed_loss)
    print("Overall accuracy on the test set:", computed_accuracy)
    print("-----------------------------------------------------")


if __name__ == '__main__':
    if '--best' in sys.argv:
        test_best_model()
    else:
        main()
