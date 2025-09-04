import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
import os

os.environ["http_proxy"] = "http://245hsbd013%40ibab.ac.in:tanega2001@proxy.ibab.ac.in:3128"
os.environ["https_proxy"] = "http://245hsbd013%40ibab.ac.in:tanega2001@proxy.ibab.ac.in:3128"

def load_data():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True,num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
    testloader = torch.utils.data.DataLoader(testset,batch_size=64,shuffle=False,num_workers=2)

    return trainloader, testloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_feature_extractor(device):
    resnet18 = models.resnet18(pretrained=True)
    for param in resnet18.parameters():
        param.requires_grad = False

    modules = list(resnet18.children())[:-1]  #removing the last fc layer
    feature_extractor = nn.Sequential(*modules)
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()  #setting to eval mode
    return feature_extractor

def extract_features(dataloader,model,device):
    features=[]
    labels=[]
    with torch.no_grad():
        loop = tqdm(dataloader, desc="Extracting features")
        for inputs, targets in loop:
            inputs = inputs.to(device)
            output = model(inputs)
            output = output.view(output.size(0), -1)
            features.append(output.cpu().numpy())
            labels.append(targets.numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

#features extracted from the CNN are fed into a classical machine learning model (SVM) which has its own separate training process
def prepare_data():
    trainloader, testloader = load_data()
    feature_extractor = get_feature_extractor(device)
    #standardizing the extracted features for SVM
    scaler = StandardScaler()
    train_features, train_labels = extract_features(trainloader, feature_extractor, device)
    test_features, test_labels = extract_features(testloader, feature_extractor, device)
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)
    return train_features, train_labels, test_features, test_labels

def main():
    print("Training SVM classifier on extracted features...")
    train_features, train_labels, test_features, test_labels = prepare_data()
    svm_classifier = SVC(kernel='linear', C=1.0)
    svm_classifier.fit(train_features, train_labels)

    print("Evaluating on test set...")
    predictions = svm_classifier.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"SVM classification accuracy on CIFAR-10 test set: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
