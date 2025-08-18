import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt

#Downloading the training data from open datasets
training_data=datasets.FashionMNIST(root='data', train=True, download=False, transform=ToTensor(),)
#Dowanloading test data from open datasets
test_data=datasets.FashionMNIST(root='data', train=False, download=False, transform=ToTensor(),)

batch_size=64 #running parallely

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# # Display image and label.
# labels_map = {
#     0: "T-Shirt",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle Boot",
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     train_features, train_labels = next(iter(train_dataloader))
#     print(f"Feature batch shape: {train_features.size()}")
#     print(f"Labels batch shape: {train_labels.size()}")
#     # img = train_features[0].squeeze()
#     # label = train_labels[0]
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()
#
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")
#
# for X, y in test_dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")#N=batch size, C=no. of channels
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)



#TRIED OUT WITHOUT CLASS

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")
#
# # Define model as a Sequential module (no class)
# model = nn.Sequential(
#     nn.Flatten(),                      # Converts [N, 1, 28, 28] → [N, 784]
#     nn.Linear(28*28, 512),             # Fully connected layer 1
#     nn.ReLU(),                         # Activation
#     nn.Linear(512, 512),               # Fully connected layer 2
#     nn.ReLU(),                         # Activation
#     nn.Linear(512, 10)                 # Output layer (10 classes)
# )
#
# # Move model to device
# model = model.to(device)
#
# # Print model architecture
# print(model)


#OPRIMIZING THE MODEL PARAMETERS

loss_fn=nn.CrossEntropyLoss()#loss function for classification problems,
# used when model outputs raw scores and targets are classic indices
optimizer=torch.optim.SGD(model.parameters(),lr=1e-3)# SGD-stochastic gradient descent optimizer
#It updates model weights using gradients from backpropagation.

def train (dataloader,model,loss_fn,optimizer):
    size=len(dataloader.dataset)
    for batch,(X,y) in enumerate(dataloader):
        X,y=X.to(device),y.to(device)

        # computing prediction error
        pred = model(X)#does a forward pass and gives predictions
        loss = loss_fn(pred, y)#returns loss value

        #Backpropagation
        loss.backward()#calculates gradients via backpropagation
        optimizer.step()#updates weights using the gradients
        optimizer.zero_grad()#clears old gradients so they don't accumulate

        #In PyTorch, most variables (inputs, weights, outputs, losses) are stored as tensors,
        # which are like NumPy arrays but with extra features like GPU support and autograd.
        if batch % 100 == 0:
            loss, current = (loss.item(),#converts tensor to python number
                             (batch + 1) * len(X))
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()#Set the module in evaluation mode.This has an effect only on certain modules.
    # See the documentation of particular modules for details of their behaviors in training/evaluation mode,
    # i.e. whether they are affected, e.g. Dropout, BatchNorm, etc.


    test_loss,correct=0,0
    with torch.no_grad():#Context-manager that disables gradient calculation.
        # Disabling gradient calculation is useful for inference, when you are sure that you will not call(reduces memory consumption)
        for X, y in dataloader:
            X, y = X.to(device),y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")#returns a dictionary containing references to the whole state of the module.
print("Saved PyTorch Model State to model.pth")

model=NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))
#If a parameter or buffer is registered as None and its corresponding key exists in state_dict, load_state_dict() will raise a RuntimeError.