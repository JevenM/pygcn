from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=250,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()




def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])

    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_train, acc_train


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return loss_test, acc_test


# 定义两个数组
Loss_train_list = []
Accuracy_list = []
Loss_test_list = []
Accuracy_test_list = []

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    loss_train, acc_train = train(epoch)
    Loss_train_list.append(loss_train.item())
    Accuracy_list.append(acc_train.item())
    if epoch % 10 == 0:
        # Testing
        loss_test, acc_test = test()
        Loss_test_list.append(loss_test.item())
        Accuracy_test_list.append(acc_test.item())

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

x = range(0, args.epochs)
y1 = Accuracy_list
y2 = Loss_train_list
x2 = range(0, int(args.epochs/10.0))
y3 = Accuracy_test_list
y4 = Loss_test_list

plt.figure(figsize=(10, 5))
plt.title('Train accuracy vs. epoches')
plt.subplot(2, 2, 1)
plt.plot(x, y1)

plt.ylabel('Train accuracy')
plt.subplot(2, 2, 2)
plt.plot(x, y2)
plt.xlabel('Train loss vs. epoches')
plt.ylabel('Train loss')
plt.subplot(2, 2, 3)
plt.plot(x2, y3)
plt.xlabel('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.subplot(2, 2, 4)
plt.plot(x2, y4)
plt.xlabel('Test loss vs. epoches')
plt.ylabel('Test loss')

plt.savefig("accuracy_loss.jpg")
plt.show()


