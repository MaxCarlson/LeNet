import torch
from torch import nn
from torchvision import datasets
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"  
#device = torch.device('cpu')
device = torch.device(dev) 

transform = transforms.Compose(
    [transforms.ToTensor()])#,
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batchSize = 128

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#activation = nn.Sigmoid()
activation = nn.Tanh()
#activation = nn.ReLU()

class Model1(nn.Module):
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

class Model2(Model1):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

class Model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=(3,3))
        self.conv2 = nn.Conv2d(6, 6, kernel_size=3, padding=(3,3))
        self.conv3 = nn.Conv2d(6, 6, kernel_size=3, padding=(3,3))
        self.conv4 = nn.Conv2d(6, 6, kernel_size=3, padding=(3,3))
        self.conv5 = nn.Conv2d(6, 6, kernel_size=3, padding=(3,3))
        self.pool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.fc1 = nn.Linear(13254, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(activation(self.conv1(x)))
        x = self.pool(activation(self.conv2(x)))
        x = self.pool(activation(self.conv3(x)))
        x = self.pool(activation(self.conv4(x)))
        x = self.pool(activation(self.conv5(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        v = x.size()
        x = activation(self.fc1(x))
        x = activation(self.fc2(x))
        x = self.fc3(x)
        return x


model = Model1()
#model = Model2()
#model = Model3()

model = model.to(device)

epochs  = 0
lr      = 0.01
#loss_fn = nn.MSELoss()
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

if __name__ == '__main__':

    trainAccPerEpoch = []
    testAccPerEpoch = []
    divergence = 0
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
                    if type(loss_fn).__name__ == 'MSELoss':
                        targets = torch.nn.functional.one_hot(targets, num_classes=len(classes))
                        targets = targets.float()
                    
                    loss = loss_fn(pred, targets)
                    if i % 100 == 99:
                        print(f'[{e + 1}, {i + 1:5d}] loss: {runningLoss / 100:.3f}')
                        runningLoss = 0.0
                    loss.backward()
                    optim.step()
                    runningLoss += loss.item()

            return numCorrect
            
        trainAcc = run(trainloader, train=True)
        testAcc = run(testloader, train=False)

        trainAcc = trainAcc / len(trainset)
        testAcc = testAcc / len(testset)

        print(f'[Train,Test] accuracy for Epoch {e+1}: {[trainAcc, testAcc]}')


        trainAccPerEpoch.append(trainAcc)
        testAccPerEpoch.append(testAcc)

        if len(testAccPerEpoch) > 2 and testAccPerEpoch[-2] >= testAccPerEpoch[-1]:
            divergence += 1
            if divergence > 3:
                break
        else:
            divergence = 0



    #plt.plot(range(len(trainAccPerEpoch)), trainAccPerEpoch, c='g')
    #plt.plot(range(len(testAccPerEpoch)), testAccPerEpoch, c='r')
    #plt.title(f'Accuracy vs. Epoch, loss = {type(loss_fn).__name__}, activation={type(activation).__name__}, lr={lr}')
    #plt.xlabel('Epoch')
    #plt.ylabel('Accuracy')
    #plt.legend(['Training Acc', 'Test Acc'])
    #plt.show()
    #
    #torch.save(model.state_dict(), './best')

    model.load_state_dict(torch.load('./best'))

    def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
        import numpy as np
        from torchvision import utils
        n,c,w,h = tensor.shape

        if allkernels: tensor = tensor.view(n*c, -1, w, h)
        elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

        rows = np.min((tensor.shape[0] // nrow + 1, 64))    
        grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
        plt.figure( figsize=(nrow,rows) )
        plt.imshow(grid.numpy().transpose((1, 2, 0)))


    filter = model.conv2.weight.clone()
    print(filter.shape)
    visTensor(filter.cpu(), ch=3, allkernels=True)

    plt.axis('off')
    plt.ioff()
    plt.show()