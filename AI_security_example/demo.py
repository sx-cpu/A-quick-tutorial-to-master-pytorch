import os 
from torchvision import datasets
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

# prepare dataset
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(root = 'pytorch_tutorial_from_Liuer/dataset/mnist',
                            train = True,
                            transform = transform,
                            download=True)
train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=batch_size)
test_data = datasets.MNIST(root = 'pytorch_tutorial_from_Liuer/dataset/mnist',
                           train = False,
                           transform = transform,
                           download = True)
test_loader = DataLoader(dataset=test_data, shuffle=True, batch_size=batch_size)

# model design
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(784, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
            # torch.nn.ReLU()
            
            )
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.dense(x)
        return x
    
model = Net()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)


# loss and optimzer
optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()


# train and test
def train(epoch):
    sum_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero grad
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        loss = loss_func(outputs, labels)

        # backward
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        if i % 100 == 99:
            print(f'epoch = {epoch+1}, batch = {i+1}, loss = {sum_loss / 100:.04f}')
            sum_loss = 0.0
            

def test(epoch):
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'epoch = {epoch+1}, accuracy = {100*correct/total:.02f}%')

if __name__ == '__main__':
    for epoch in range(20):
        train(epoch)
        test(epoch)
            



