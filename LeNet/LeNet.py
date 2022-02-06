import torch
from torch import nn
from torchvision import datasets
import torchvision.transforms as transforms

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"  
#device = torch.device('cpu')
device = torch.device(dev) 

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batchSize = 4

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#activation = nn.Sigmoid()
#activation = nn.Tanh()
activation = nn.ReLU()

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(activation(self.conv1(x)))
        x = self.pool(activation(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = activation(self.fc1(x))
        x = activation(self.fc2(x))
        x = self.fc3(x)
        return x

#model = nn.Sequential(
#    nn.Conv2d(3, 6, kernel_size=5), activation,
#    nn.AvgPool2d(kernel_size=2, stride=2),
#    nn.Conv2d(6, 16, kernel_size=5), activation,
#    nn.AvgPool2d(kernel_size=2, stride=2),
#    nn.Flatten(),
#    nn.Linear(16 * 5 * 5, 120), activation,
#    nn.Linear(120, 84), activation,
#    nn.Linear(84, len(classes))
#    )
model = Model()
model = model.to(device)

epochs  = 10
lr      = 0.001
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)



if __name__ == '__main__':

    trainAccPerEpoch = []
    testAccPerEpoch = []
    for e in range(epochs):
        def run(loader, train):
            numCorrect = 0
            runningLoss = 0.0
            for i, (features, targets) in enumerate(loader, 0):
    
                features    = features.to(device)
                targets     = targets.to(device)

                if dev == 'cpu':
                    x = features.numpy()
                
                pred = model(features)
                
                if dev == 'cpu':
                    p = pred.detach().numpy()
                    t = targets.numpy()

                # Calculate number of correct values
                maxVals, maxIdx = torch.max(pred, 1)
                numCorrect += (maxIdx == targets).sum().item()

                if train:
                    optim.zero_grad()
                    loss = loss_fn(pred, targets)
                    if i % 2000 == 1999:    # print every 2000 mini-batches
                        print(f'[{e + 1}, {i + 1:5d}] loss: {runningLoss / 2000:.3f}')
                        runningLoss = 0.0
                    loss.backward()
                    optim.step()
                    runningLoss += loss.item()
                #else:
                    # Calculate number of correct values
                    #maxVals, maxIdx = torch.max(pred, 1)
                    #numCorrect += (maxIdx == targets).sum().item()
            return numCorrect
            
        trainAcc = run(trainloader, train=True)
        testAcc = run(testloader, train=False)

        trainAcc = trainAcc / len(trainset)
        testAcc = testAcc / len(testset)

        print(f'[Train,Test] accuracy for Epoch {e+1}: {[trainAcc, testAcc]}')
        trainAccPerEpoch.append(trainAcc)
        testAccPerEpoch.append(testAcc)


    plt.plot(range(len(trainAccPerEpoch)), trainAccPerEpoch, c='g')
    plt.plot(range(len(testAccPerEpoch)), testAccPerEpoch, c='r')
    plt.title(f'Accuracy vs. Epoch, 2 x 1024 FC layers, ReLU, lr={lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training Acc', 'Test Acc'])
    plt.show()