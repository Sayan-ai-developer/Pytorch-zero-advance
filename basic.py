import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the model
class NN:
    def __init__(self, features:int) -> None:
        self.weights = torch.rand(features, 1, device=device, dtype=torch.float32, requires_grad=True) 
        self.bias = torch.rand(1, device=device ,dtype=torch.float32, requires_grad=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_pred = torch.sigmoid(torch.mm(x, self.weights) + self.bias)
        
        return y_pred
    
    def loss_function(self, y_pred:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        loss = -torch.mean(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred))
        
        return loss

# Defining the hyperparameters
lr = float(input("Enter the learning rate: "))
epochs = int(input("Enter the number of epochs: "))

# Defining the model
model =  NN(2)

# Defining the input and output
x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=device, dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], device=device, dtype=torch.float32)      

# Training the model
for epoch in range(epochs):
    y_pred = model.forward(x)
    loss = model.loss_function(y_pred, y)
    loss.backward()
    
    with torch.no_grad():
        model.weights -= lr * model.weights.grad
        model.bias -= lr * model.bias.grad
        
    model.weights.grad.zero_()
    model.bias.grad.zero_()
    
    print(f'Epoch: {epoch+1}/{epochs}, Loss: {loss.item()}')









