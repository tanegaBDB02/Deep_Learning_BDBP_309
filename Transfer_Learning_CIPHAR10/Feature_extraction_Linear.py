import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
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

def feature_extractor(device):
    resnet18 = models.resnet18(pretrained=True)
    for param in resnet18.parameters():
        param.requires_grad = False

    #replacing final FC layer for CIFAR-10
    num_features = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_features, 10)
    resnet18 = resnet18.to(device)

    return resnet18

def train_loop(trainloader,model,criterion,optimizer,device,num_epochs=5):
    for epoch in range(5):  #train for 5 epochs
        running_loss = 0.0
        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}")

def test_loop(testloader,model,device):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy on CIFAR-10 test set: {100 * correct / total:.2f}%")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = load_data()
    model = feature_extractor(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(),lr=0.001,momentum=0.9)
    train_loop(trainloader,model,criterion,optimizer,device)
    test_loop(testloader,model,device)

if __name__ == "__main__":
    main()
