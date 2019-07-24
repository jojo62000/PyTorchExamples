
#Import required libraries
import torch as tch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix

#Create 500 observations with randn | This will be tagged as 0
X1 = tch.randn(3000, 32)
#Create another 500 observations with randn slightly different from X1| This will be tagged as 0
X2 = tch.randn(3000, 32) + 0.5
#Ccombined X1 and X2
X = tch.cat([X1, X2], dim=0)

#Create 1000 Y combined 50% 0's and 50% 1's
Y1 = tch.zeros(3000, 1)
Y2 = tch.ones(3000, 1)
Y = tch.cat([Y1, Y2], dim=0)

# Creating data indices for training and validation splits:
batch_size = 16
validation_split = 0.2 # 20%
random_seed= 2019

#Shuffle indices
dataset_size = X.shape[0]
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.seed(random_seed)
np.random.shuffle(indices)

#Create train and validation indices
train_indices, val_indices = indices[split:], indices[:split]
#Create train and validation dataset
X_train, x_test = X[train_indices], X[val_indices]
Y_train, y_test = Y[train_indices], Y[val_indices]

#Print shapes of each dataset
print("X_train.shape:",X_train.shape)
print("x_test.shape:",x_test.shape)
print("Y_train.shape:",Y_train.shape)
print("y_test.shape:",y_test.shape)


#Define a neural network with 2 hidden layers and 1 output layer
#Hidden Layers will have 64 and 256 neurons
#Output layers will have 1 neuron

class NeuralNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 256)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(256, 1)
        self.final = nn.Sigmoid()
        
    def forward(self, x):
        op = self.fc1(x)
        op = self.relu1(op)        
        op = self.fc2(op)
        op = self.relu2(op)
        op = self.out(op)
        y = self.final(op)
        return y
    

model = NeuralNetwork()
loss_function = nn.BCELoss()  #Binary Crosss Entropy Loss
optimizer = tch.optim.Adam(model.parameters(),lr= 0.001)

num_epochs = 10
batch_size=16

for epoch in range(num_epochs):
    train_loss= 0.0

    #Explicitly start model training
    model.train()
    
    for i in range(0,X_train.shape[0],batch_size):

        #Extract train batch from X and Y
        input_data = X_train[i:min(X_train.shape[0],i+batch_size)]
        labels = Y_train[i:min(X_train.shape[0],i+batch_size)]
        
        #set the gradients to zero before starting to do backpropragation 
        optimizer.zero_grad()
        
        #Forward pass
        output_data  = model(input_data)
        
        #Caculate loss
        loss = loss_function(output_data, labels)
        
        #Backpropogate
        loss.backward()
        
        #Update weights
        optimizer.step()
        
        train_loss += loss.item() * batch_size

    print("Epoch: {} - Loss:{:.4f}".format(epoch+1,train_loss/X_train.shape[0] ))

#Predict
y_test_pred = model(x_test)
a =np.where(y_test_pred>0.5,1,0)
confusion_matrix(y_test,a)

