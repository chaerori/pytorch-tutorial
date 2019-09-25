import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# design neural network with pytorch module
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    # Must implement forward() for every subclass of nn.Module
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

# create model with defined neuralnet module
model = NeuralNet(100, 50, 100)
print("\nModel Structure: ")
print(model)

# make dummy input & target data
input_data = torch.randn(100, requires_grad=True)
target_data = torch.full((100,), 30, requires_grad=False) # ground truth

# forward propagation(the two lines below are functionally identical)
#output_data = model.forward(input_data)
output_data = model(input_data)

# define loss function
criterion = nn.MSELoss()
loss = criterion(output_data, target_data)
print("Loss: %F" %loss.item())

optimizer = optim.SGD(model.parameters(), lr=0.01)

print(model.input_layer.weight)

# calculate gradient on autograd backpropagation
# if you call backward(), you will get accumulated gradient for all data in graph
# so you have to call zero_grad() before calling backward() to erase all buffered data
model.zero_grad()
optimizer.zero_grad()
loss.backward()
# update the parameters in model
optimizer.step()

print(model.input_layer.weight)