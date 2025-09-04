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
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    return trainloader, testloader

def fine_tune_partial(model):
    #layers to freeze (initial layers)
    freeze_layers = ['conv1','bn1','layer1','layer2']
    #layers to unfreeze (later layers)
    unfreeze_layers = ['layer3','layer4','fc']

    for name, module in model.named_children():
        if name in freeze_layers:
            for param in module.parameters():
                param.requires_grad = False
        elif name in unfreeze_layers:
            for param in module.parameters():
                param.requires_grad = True

def train_loop(trainloader, model, criterion, optimizer, device,num_epochs=5):
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images) #forward pass
            loss = criterion(outputs, labels) #calculate loss
            loss.backward() #backward pass
            optimizer.step() #update loss
            running_loss=running_loss+loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


def test_loop(testloader, model, device):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1) #inds the index of the highest score in each row.
            total += labels.size(0)
            correct += (predicted == labels).sum().item() #how many predictions were correct in this batch
    print(f"Accuracy on CIFAR-10 test set: {100 * correct / total:.2f}%")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = load_data()

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model = model.to(device)
    fine_tune_partial(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=0.001, momentum=0.9)
    train_loop(trainloader, model, criterion, optimizer, device, num_epochs=5)
    test_loop(testloader, model, device)

if __name__ == "__main__":
    main()
