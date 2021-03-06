import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tangent_images.util import BrownDistortion

import os
import visdom

# Parameters of training
n_epochs = 20
batch_size_train = 32
batch_size_test = 32
learning_rate = 5e-3
log_interval = 40
device_id = 0

# Data params
data_mu = 0.1307
data_std = 0.3081
K1 = 0.6
K2 = 0.0
saved_model = 'model.pth'  # Set to None to retrain

# Repeatability
random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)

os.makedirs('results', exist_ok=True)

if saved_model is None:
    vis = visdom.Visdom()
    name = 'distorted-mnist'

    vis.line(
        env=name,
        X=torch.zeros(1, 1).long(),
        Y=torch.zeros(1, 1).float(),
        win='training_loss',
        opts=dict(
            title='Training Loss',
            markers=False,
            xlabel='Iteration',
            ylabel='Loss',
            legend=['Train Loss']))

    vis.line(
        env=name,
        X=torch.zeros(1, 1).long(),
        Y=torch.zeros(1, 1).float(),
        win='testing_loss',
        opts=dict(
            title='Test Loss',
            markers=False,
            xlabel='Epoch',
            ylabel='Loss',
            legend=['Test Loss']))

    vis.line(
        env=name,
        X=torch.zeros(1, 1).long(),
        Y=torch.zeros(1, 1).float(),
        win='accuracy',
        opts=dict(
            title='Accuracy',
            markers=False,
            xlabel='Epoch',
            ylabel='Accuracy',
            legend=['Accuracy']))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32 * 7 * 7, 10, kernel_size=1)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d((self.conv2(x)), 2))
        x = x.view(-1, 32 * 7 * 7, 1, 1)
        x = self.conv3(x)
        x.squeeze_()
        return F.log_softmax(x, dim=-1)


# Initialize network and optimizer
device = 'cpu' if not torch.cuda.is_available() else torch.device('cuda', 0)
network = Net()
network = network.to(device)
if saved_model is not None:
    network.load_state_dict(torch.load(saved_model))
optimizer = optim.Adam(network.parameters(), lr=learning_rate)

# Progress trackers
train_losses = AverageMeter()
train_counter = []
test_losses = []


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        # print(data.shape)
        target = target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            train_losses.update(loss.item())
            train_counter.append((batch_idx * batch_size_train) + (
                (epoch - 1) * len(train_loader.dataset)))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_losses.avg))
            torch.save(network.state_dict(), 'results/model.pth')
            torch.save(optimizer.state_dict(), 'results/optimizer.pth')

            total_num_batches = (epoch - 1) * len(train_loader) + batch_idx

            vis.line(
                env=name,
                X=torch.tensor([total_num_batches]),
                Y=torch.tensor([train_losses.avg]),
                win='training_loss',
                update='append',
                opts=dict(legend=['Train Loss']))


def test(epoch):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if saved_model is None:
        vis.line(
            env=name,
            X=torch.tensor([epoch]),
            Y=torch.tensor([test_loss]),
            win='testing_loss',
            update='append',
            opts=dict(legend=['Test Loss']))
        vis.line(
            env=name,
            X=torch.tensor([epoch]),
            Y=torch.tensor([100. * correct / len(test_loader.dataset)]),
            win='accuracy',
            update='append',
            opts=dict(legend=['Accuracy']))
        vis.images(
            data * data_std + data_mu,
            env=name,
            win='rgb',
            opts=dict(title='Test Images', caption=''))

    # Return accuracy
    return 100. * correct / len(test_loader.dataset)


# Dataloaders
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        'files/',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((data_mu, ), (data_std, )),
        ])),
    batch_size=batch_size_train,
    shuffle=True)

# If no prior saved model, train the network from scratch
if saved_model is None:
    test(0)
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test(epoch)

# If using a saved model, just run evaluation
else:
    with open('accuracy.txt', 'w') as f:
        for K1 in range(51):
            print('K1:', K1 / 100)
            test_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(
                    'files/',
                    train=False,
                    download=True,
                    transform=torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        BrownDistortion(K1 / 100, K2, 0, 0, 0),
                        torchvision.transforms.Normalize((data_mu, ),
                                                         (data_std, )),
                    ])),
                batch_size=batch_size_test,
                shuffle=True)
            accuracy = test(0)
            f.write('{:0.2f} {:0.2f}\n'.format(K1, accuracy))