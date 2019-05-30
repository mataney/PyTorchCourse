import torch
import torchvision

import torchvision.transforms as transforms
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter


train_writer = SummaryWriter('tensorboard_logs/CIFAR10-2/train')
val_writer = SummaryWriter('tensorboard_logs/CIFAR10-2/val')
device = torch.device('cpu')


transform = transforms.Compose([transforms.ToTensor()])

batch_size = 32
report_every = 100
learning_rate = 0.001
momentum = 0.9
num_epochs = 2

###### <tensorboard> ######
train_writer.add_text('Hyperparameters', f"batch_size: {batch_size}, report_every: {report_every}, learning_rate: {learning_rate}, momentum: {momentum}", 0)
###### </tensorboard> ######

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        batch, channels, w, h = x.size()        
        x = self.linear(x.view(batch, -1))
        return x

class FC(nn.Module):
  def __init__(self, in_size1, in_size2, in_size3, out_size, drop_prob):
    super(FC, self).__init__()
    self.linear1 = nn.Linear(in_size1, in_size2)
    self.linear2 = nn.Linear(in_size2, in_size3)
    self.out_linear = nn.Linear(in_size3, out_size)
    
    self.drop = nn.Dropout(drop_prob)
    self.drop2 = nn.Dropout(drop_prob) # This is not really necessary, it's just for better tensorboarding
      
  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = self.drop(x)
    x = self.linear1(x)
    x = F.relu(x)
    x = self.linear2(x)
    x = F.relu(x)
    x = self.drop2(x)
    x = self.out_linear(x)
    
    return x

model = FC(3 * 32 * 32, 1024, 512, 10, 0.1)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

def evaluate(model, dataloader, device, epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    ###### </tensorboard> ######
    val_writer.add_scalar('Accuracy', 100 * correct / total, epoch)
    ###### <tensorboard> ######


def train(model, num_epochs, trainloader, optimizer, criterion, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            if i > 1000:
                break
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            ###### <tensorboard> ######
            if i == 0: 
                grid = torchvision.utils.make_grid(inputs)
                train_writer.add_image('images', grid, 0)
                train_writer.add_graph(model, inputs)
            ###### </tensorboard> ######

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % report_every == report_every-1:    # print every report_every mini-batches
            #   print('[%d, %5d] loss: %.3f' %
                    # (epoch + 1, i + 1, running_loss / report_every))
                ###### <tensorboard> ######
                train_writer.add_scalar('loss', running_loss / report_every, epoch * len(trainloader) + i + 1)
                for name, param in model.named_parameters():
                    train_writer.add_histogram(name, param.clone().cpu().data.numpy(), 0)
                ###### </tensorboard> ######
                running_loss = 0.0

        evaluate(model, testloader, device, epoch)

train(model, num_epochs, trainloader, optimizer, criterion, device)

images = torch.stack([t[0] for i, t in enumerate(testset) if i < 100])
label = torch.Tensor([t[1] for i, t in enumerate(testset) if i < 100])

###### <tensorboard> ######
val_writer.add_embedding(images.view(100, -1), metadata=label, label_img=images)
###### </tensorboard> ######