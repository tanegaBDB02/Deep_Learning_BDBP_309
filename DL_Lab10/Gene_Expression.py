import torch.optim as optim
from torch import nn
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset,DataLoader
from DL_Lab10.Preprocess import load_data
from sklearn.preprocessing import StandardScaler

def data_split():
    full_data=load_data("/home/ibab/Deep_Learning/DL_Lab10/Landmark_genes.txt",
                        "/home/ibab/Deep_Learning/DL_Lab10/Target_genes.txt").astype(np.float32)
    X_train,X_temp=train_test_split(full_data,test_size=0.2,random_state=42)
    X_val,X_test=train_test_split(X_temp,test_size=0.5,random_state=42)

    scaler = StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_val_scaled=scaler.transform(X_val)
    X_test_scaled=scaler.transform(X_test)

    print("Training set:",X_train_scaled.shape,"Validation set:",X_val_scaled.shape,"Testing set:",X_test_scaled.shape)
    return X_train_scaled,X_val_scaled,X_test_scaled

def make_loaders(X_train_scaled,X_val_scaled,X_test_scaled,batch_size=64):
    train=TensorDataset(torch.tensor(X_train_scaled),torch.tensor(X_train_scaled))
    val=TensorDataset(torch.tensor(X_val_scaled),torch.tensor(X_val_scaled))
    test=TensorDataset(torch.tensor(X_test_scaled),torch.tensor(X_test_scaled))

    train_loader=DataLoader(train,batch_size=batch_size,shuffle=True)
    val_loader=DataLoader(val,batch_size=batch_size,shuffle=False)
    test_loader=DataLoader(test,batch_size=batch_size,shuffle=False)

    return train_loader,val_loader,test_loader

class NeuralNetwork(nn.Module):
    def __init__(self,input_dimension):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Linear(input_dimension,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024,input_dimension),
        )

    def forward(self,x):
        x=self.encoder(x)
        logits=self.decoder(x)
        return logits

def train_loop(model,dataloader,criterion,optimizer):
    model.train()
    train_loss=0
    size=len(dataloader)
    for X,_ in dataloader:
        optimizer.zero_grad()
        output=model(X)
        loss=criterion(output,X)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
    return train_loss/size

def eval_loop(model,dataloader,criterion):
    model.eval()
    size=len(dataloader)
    total_loss=0
    with torch.no_grad():
        for X,_ in dataloader:
            pred=model(X)
            test_loss=criterion(pred,X)
            total_loss=total_loss+test_loss.item()
    return total_loss/size

def main():
    X_train_scaled,X_val_scaled,X_test_scaled=data_split()
    train_loader,val_loader,test_loader=make_loaders(X_train_scaled,X_val_scaled,X_test_scaled)

    input_dim=X_train_scaled.shape[1]
    model=NeuralNetwork(input_dim)
    criterion=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),lr=1e-3)

    epochs=100
    for epoch in range(epochs):
        train_loss=train_loop(model, train_loader, criterion, optimizer)
        val_loss=eval_loop(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    test_loss = eval_loop(model, test_loader, criterion)
    print("\nFinal Test Loss:", test_loss)


if __name__=="__main__":
    main()