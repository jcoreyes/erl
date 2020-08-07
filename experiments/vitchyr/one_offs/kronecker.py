from __future__ import print_function
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
from rlkit.pythonplusplus import line_logger
from rlkit.torch.pytorch_util import kronecker_product

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class KroneckerLinear(nn.Module):
    def __init__(self):
        super(KroneckerLinear, self).__init__()
        self.w1 = nn.Parameter(torch.Tensor(2, 5))
        self.w2 = nn.Parameter(torch.Tensor(5, 2))
        self.w3 = nn.Parameter(torch.Tensor(2, 1))
        self.w4 = nn.Parameter(torch.Tensor(2, 1))
        self.w5 = nn.Parameter(torch.Tensor(2, 1))
        self.w6 = nn.Parameter(torch.Tensor(2, 1))
        self.w7 = nn.Parameter(torch.Tensor(2, 1))
        self.bias = nn.Parameter(torch.Tensor(10))
        self.reset_parameters()

    def reset_parameters(self):
        for w in [
            self.w1,
            self.w2,
            self.w3,
            self.w4,
            self.w5,
            self.w6,
            self.w7,
        ]:
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        w = kronecker_product(self.w1, self.w2)
        w = kronecker_product(w, self.w3)
        w = kronecker_product(w, self.w4)
        w = kronecker_product(w, self.w5)
        w = kronecker_product(w, self.w6)
        w = kronecker_product(w, self.w7)
        batch_size = input.size()[0]
        return input @ w + self.bias.repeat(batch_size, 1)

    def __repr__(self):
        return "{0} ({1} -> {2})".format(
            self.__class__.__name__,
            320,
            50,
        )


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc1 = nn.Linear(320, 10)
        self.fc1 = KroneckerLinear()
        # self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.fc1(x)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        return F.log_softmax(x)

from operator import mul
from functools import reduce
def product(x):
    return reduce(mul, x)

model = Net()
if args.cuda:
    model.to(ptu.device)
params = list(model.parameters())
print(sum([product(p.size()) for p in params]))

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.to(ptu.device), target.to(ptu.device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            line_logger.print_over('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.to(ptu.device), target.to(ptu.device)
        data, target = Variable(data, requires_grad=False), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    line_logger.newline()
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
