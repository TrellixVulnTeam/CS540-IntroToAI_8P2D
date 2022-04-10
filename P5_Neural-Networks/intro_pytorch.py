import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Building a Deep-Learning Model for Predicting Labels of Hand Written Images, using the Fashion-MNIST dataset

# get_data_loader(): inputs an optional boolean argument (default value is True for training dataset), and returns a Dataloader
#  (torchvision.datasets.FashionMNIST object) for the training set (if training = True) or the test set (if training = False)
def get_data_loader(training = True):
# Input preprocessing: Specifying Transform
    custom_transform= transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    if(training == True):
         # The TRAIN SET contains images and labels to train the neural network
        train_set = datasets.FashionMNIST('./data',train=True, download=True,transform=custom_transform)
        loader = torch.utils.data.DataLoader(train_set, batch_size=64)

    else:
        # The TEST SET contains images and labels for model evaluation
        test_set = datasets.FashionMNIST('./data', train=False, transform=custom_transform)
        loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    return loader



def build_model():
    # Neural Network Layers to ImplementS
    # 1. A Flatten layer to convert the 2D pixel array to a 1D array.
    # 2. A Dense layer with 128 nodes and a ReLU activation.
    # 3. A Dense layer with 64 nodes and a ReLU activation.
    # 4. A Dense layer with 10 nodes.
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=784, out_features=128, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=64, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=10, bias=True)
    )
    return model

def train_model(model, train_loader, criterion, T):
    # Put the Model in Training Mode
    model.train()
    # Compute the train DataLoader length
    loader_len = len(train_loader.dataset)  # EDIT 1
    # Define the Standard Gradient Decent Optimizer
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(T):  # loop over the dataset multiple times
        epoch_loss = 0.0
        predicted = 0
        total = 0
        correct = 0
        for i, (images, labels) in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # images, labels = data
            # zero the parameter gradients
            opt.zero_grad()
            # forward + backward + optimize
            y_predict = model(images)
            loss = criterion(y_predict, labels)
            loss.backward()
            opt.step()

            # Compute Average
            _, predicted = torch.max(y_predict.data, dim = 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Compute Total Loss for the Current Epoch
            epoch_loss += loss.item() * 64 # EDIT 2
        # Compute the Accumulated Loss (Epoch Loss / Length of Dataset) 
        print("Train Epoch: " + str(epoch), " Accuracy: " + str(correct) + "/" + str(total) + "(" + str("{:.2%}".format(correct/total)) + ")", " Loss: " + "{:.3f}".format(epoch_loss/loader_len))
        epoch_loss = 0.0

def evaluate_model(model, test_loader, criterion, show_loss = True):
    # Put model in Evaluation Mode
    model.eval()
     # Compute the train DataLoader length
    loader_len = len(test_loader.dataset) # EDIT 3
    # Define the Standard Gradient Decent Optimizer
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    with torch.no_grad():
        predicted = 0
        total = 0
        correct = 0
        total_loss = 0
        for i, (images, labels) in enumerate(test_loader, 0):
            y_predict = model(images)
            loss = criterion(y_predict, labels)
            # Compute Average
            _, predicted = torch.max(y_predict.data, dim = 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item() * 64 # EDIT 4
        if(show_loss == True):
            print("Average loss: " + str(format(total_loss/total, ".4f")))
        print("Accuracy: " + str("{:.2%}".format(correct/total)))

def predict_label(model, test_images, index):
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    len_classes = len(class_names)
    images, labels = next(iter(test_images))
    test_image = images[index]
    y_predict = model(test_image)
    probs = F.softmax(y_predict, dim = 1)
    percentage_probs = np.zeros(len_classes)
    for i in range(len_classes):
        percentage_probs[i] = (probs[0][i]/1)
    largest_idxs = (-percentage_probs).argsort()[:3]
    for i in range(3):
        print(class_names[largest_idxs[i]] + ": " + str("{:.2%}".format(percentage_probs[largest_idxs[i]])))

#  Source -> https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#iterate-through-the-dataloader + Peter Bryant
# def visualize_image(data_loader, index):
#     class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
#     images, labels = next(iter(data_loader))
#     image = images[index].squeeze()
#     label = labels[index]
#     plt.imshow(image, cmap="gray")
#     plt.savefig("image.png")
#     print("Ground Truth Label: " + class_names[label])

# if __name__ == '__main__':
#     # 1. get_data_loader()
#     train_loader = get_data_loader()
#     print(type(train_loader))
#     print(train_loader.dataset)
#     test_loader = get_data_loader(False)

#     # 2. build_model()
#     model = build_model()
#     print(model)

#     # 3. train_model()
#     criterion = nn.CrossEntropyLoss()
#     train_model(model, train_loader, criterion, T = 5)
#     # train_model(model, train_loader, criterion, T = 1)

#     # 4. evaluate_model()
#     # evaluate_model(model, test_loader, criterion, show_loss = False)
#     evaluate_model(model, test_loader, criterion, show_loss = True)

#     # 5. predict_label()
#     predict_label(model, test_loader, 1)

#     # Testing - visualize_image()
#     # visualize_image(tessst_loader, 1)
#     pass

