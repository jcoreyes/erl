"""
Script based on:
https://github.com/pytorch/examples/blob/master/mnist/main.py

Selu implementation from:
https://github.com/dannysdeng/selu/blob/master/selu.py

Summary of findings (using default argparse parameters:

Dropout | Non-linearity | Test accuracy
---------------------------------------
Normal  | relu          | 11
Normal  | selu          | 47
Normal  | tanh          | 11
Normal  | 1.6 * tanh    | 40
Alpha   | relu          | 11
Alpha   | selu          | 98
Alpha   | tanh          | 45
Alpha   | 1.6 * tanh    | 98 (seems to learn a tad faster than selu)
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
from rlkit.pythonplusplus import line_logger

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


class selu(nn.Module):
    def __init__(self):
        super(selu, self).__init__()
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def forward(self, x):
        return self.scale * (
            F.relu(x) + self.alpha * (F.elu(-1 * F.relu(-1 * x)))
        )
        # temp1 = self.scale * F.relu(x)
        # temp2 = self.scale * self.alpha * (F.elu(-1 * F.relu(-1 * x)))
        # return temp1 + temp2


class alpha_drop(nn.Module):
    def __init__(self, p=0.05, alpha=-1.7580993408473766, fixedPointMean=0,
                 fixedPointVar=1):
        super(alpha_drop, self).__init__()
        keep_prob = 1 - p
        self.a = np.sqrt(fixedPointVar / (keep_prob * (
        (1 - keep_prob) * pow(alpha - fixedPointMean, 2) + fixedPointVar)))
        self.b = fixedPointMean - self.a * (
        keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        self.alpha = alpha
        self.keep_prob = 1 - p
        self.drop_prob = p
        self.selu = selu()

    def forward(self, x):
        if self.keep_prob == 1 or not self.training:
            # print("testing mode, direct return")
            return x
        else:
            random_tensor = self.keep_prob + torch.rand(x.size())
            binary_tensor = Variable(torch.floor(random_tensor).to(ptu.device))
            x = x.mul(binary_tensor)
            ret = x + self.alpha * (1 - binary_tensor)
            ret.mul_(self.a).add_(self.b)
            return ret


class dropout(nn.Module):
    def forward(self, x):
        return F.dropout(x, training=self.training)




class DeepSNTanhNet(nn.Module):
    """
    ~98% test accuracy with alpha dropout
    ~40% test accuracy with dropout
    """
    def __init__(self):
        super(DeepSNTanhNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 50)
        self.fc6 = nn.Linear(50, 50)
        self.fc7 = nn.Linear(50, 50)
        self.fc8 = nn.Linear(50, 50)
        self.fc9 = nn.Linear(50, 10)
        self.dropout1 = create_drop()
        self.dropout2 = create_drop()
        self.dropout3 = create_drop()
        self.dropout4 = create_drop()
        self.dropout5 = create_drop()
        self.dropout6 = create_drop()
        self.dropout7 = create_drop()
        self.dropout8 = create_drop()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.tanh(self.fc1(x)) * 1.6
        x = self.dropout1(x)
        x = F.tanh(self.fc2(x)) * 1.6
        x = self.dropout2(x)
        x = F.tanh(self.fc3(x)) * 1.6
        x = self.dropout3(x)
        x = F.tanh(self.fc4(x)) * 1.6
        x = self.dropout5(x)
        x = F.tanh(self.fc5(x)) * 1.6
        x = self.dropout5(x)
        x = F.tanh(self.fc6(x)) * 1.6
        x = self.dropout6(x)
        x = F.tanh(self.fc7(x)) * 1.6
        x = self.dropout7(x)
        x = F.tanh(self.fc8(x)) * 1.6
        x = self.dropout8(x)
        x = self.fc9(x)
        return F.log_softmax(x)


class DeepTanhNet(nn.Module):
    """
    ~11% test accuracy with alpha dropout
    ~45% test accuracy with dropout
    """
    def __init__(self):
        super(DeepTanhNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 50)
        self.fc6 = nn.Linear(50, 50)
        self.fc7 = nn.Linear(50, 50)
        self.fc8 = nn.Linear(50, 50)
        self.fc9 = nn.Linear(50, 10)
        self.dropout1 = create_drop()
        self.dropout2 = create_drop()
        self.dropout3 = create_drop()
        self.dropout4 = create_drop()
        self.dropout5 = create_drop()
        self.dropout6 = create_drop()
        self.dropout7 = create_drop()
        self.dropout8 = create_drop()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.tanh(self.fc1(x))
        x = self.dropout1(x)
        x = F.tanh(self.fc2(x))
        x = self.dropout2(x)
        x = F.tanh(self.fc3(x))
        x = self.dropout3(x)
        x = F.tanh(self.fc4(x))
        x = self.dropout5(x)
        x = F.tanh(self.fc5(x))
        x = self.dropout5(x)
        x = F.tanh(self.fc6(x))
        x = self.dropout6(x)
        x = F.tanh(self.fc7(x))
        x = self.dropout7(x)
        x = F.tanh(self.fc8(x))
        x = self.dropout8(x)
        x = self.fc9(x)
        return F.log_softmax(x)


class DeepSeluNet(nn.Module):
    """
    ~98% test accuracy with alpha dropout
    ~47% test accuracy with dropout
    """
    def __init__(self):
        super(DeepSeluNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 50)
        self.fc6 = nn.Linear(50, 50)
        self.fc7 = nn.Linear(50, 50)
        self.fc8 = nn.Linear(50, 50)
        self.fc9 = nn.Linear(50, 10)
        self.dropout1 = create_drop()
        self.dropout2 = create_drop()
        self.dropout3 = create_drop()
        self.dropout4 = create_drop()
        self.dropout5 = create_drop()
        self.dropout6 = create_drop()
        self.dropout7 = create_drop()
        self.dropout8 = create_drop()
        self.selu = selu()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.selu(self.fc1(x))
        x = self.dropout1(x)
        x = self.selu(self.fc2(x))
        x = self.dropout2(x)
        x = self.selu(self.fc3(x))
        x = self.dropout3(x)
        x = self.selu(self.fc4(x))
        x = self.dropout5(x)
        x = self.selu(self.fc5(x))
        x = self.dropout5(x)
        x = self.selu(self.fc6(x))
        x = self.dropout6(x)
        x = self.selu(self.fc7(x))
        x = self.dropout7(x)
        x = self.selu(self.fc8(x))
        x = self.dropout8(x)
        x = self.fc9(x)
        return F.log_softmax(x)


class DeepReluNet(nn.Module):
    """
    ~11% test accuracy with alpha dropout
    ~11% test accuracy with dropout
    """
    def __init__(self):
        super(DeepReluNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 50)
        self.fc6 = nn.Linear(50, 50)
        self.fc7 = nn.Linear(50, 50)
        self.fc8 = nn.Linear(50, 50)
        self.fc9 = nn.Linear(50, 10)
        self.dropout1 = create_drop()
        self.dropout2 = create_drop()
        self.dropout3 = create_drop()
        self.dropout4 = create_drop()
        self.dropout5 = create_drop()
        self.dropout6 = create_drop()
        self.dropout7 = create_drop()
        self.dropout8 = create_drop()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        x = F.relu(self.fc5(x))
        x = self.dropout5(x)
        x = F.relu(self.fc6(x))
        x = self.dropout6(x)
        x = F.relu(self.fc7(x))
        x = self.dropout7(x)
        x = F.relu(self.fc8(x))
        x = self.dropout8(x)
        x = self.fc9(x)
        return F.log_softmax(x)


def create_drop():
    return alpha_drop()
    # return dropout()


# model = DeepReluNet()
model = DeepSeluNet()
# model = DeepTanhNet()
# model = DeepSNTanhNet()
if args.cuda:
    model.to(ptu.device)

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
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(
        test_loader)  # loss function already averages over batch size
    line_logger.newline()
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
