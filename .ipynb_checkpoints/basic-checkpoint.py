import torch
from torch import nn
import torch.nn.functional  as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='/cnn-data',train = True, download = True, transform=transform)

test_data = datasets.MNIST(root='/cnn-data',train = False, transform=transform)

train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)

test_loader = DataLoader(test_data, batch_size = 64, shuffle = True)

class cnnModel(nn.Module):
    def __init__(self, input_shape:int, ouput_shape:int, hidden_shape:int):
        super(cnnModel).___init__()
        
        self.conv1 = nn.Conv2d(input_shape,6,3,1)
        self.conv2 = nn.Conv2d(6, 15, 3,1)
        
        self.flatten = nn.Flatten()
        
        self.dense1 = nn.Linear(5*5*15,hidden_shape)
        self.dense2 = nn.Linear(hidden_shape,ouput_shape)
        
    def forward(self, x:torch.Tensor):
        
        x = self.conv1(x)
        x = F.max_pool2d(x,2,2)
        x = self.conv2(x)
        x = F.max_pool2d(x,2,2)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        
        return F.log_softmax(x, dim = 1)
    
    
torch.manual_seed(41)

model = cnnModel(1,10,120)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

compile_model = torch.compile(model)

epochs = 100

for i in range(epochs):
    for x_train,y_train in train_loader:
        y_pred = compile_model(x_train)
        loss = loss_func(y_pred,y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()