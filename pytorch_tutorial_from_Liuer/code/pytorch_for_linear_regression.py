# -------------------------------------------------------------------------
# Pytorch Fashion
# 1. Prepare dataset
# 2. Design model using class (inherit from nn.Module)
# 3. Construct loss and optimizer (using Pytorch API)
# 4. Training cycle (forward backward update)
# -------------------------------------------------------------------------

# In pytorch, the computational graph is in mini-batch fashion, so X and Y are 3 * 1 Tensors


import torch

## prepare dataset
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

# design model using class 
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

# Construct loss and optimizer
criterion = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training cycle
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'w = {model.linear.weight.item()}')
print(f'b = {model.linear.bias.item()}')

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print(f'y_pred {y_test.data}')