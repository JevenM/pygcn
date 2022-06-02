from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data_federated, accuracy
from pygcn.models import GCN

import syft as sy  # <-- NEW: import the Pysyft library
hook = sy.TorchHook(torch)  # <-- NEW: hook PyTorch ie add extra functionalities to support Federated Learning
node1 = sy.VirtualWorker(hook, id="node1")  # <-- NEW: define remote worker bob
node2 = sy.VirtualWorker(hook, id="node2")  # <-- NEW: and alice
secure_worker = sy.VirtualWorker(hook, id="secure_worker")

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
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

# Load data for federated learning
adj_, features_, labels_, idx_train1, idx_val1, idx_test, idx_train2, idx_val2 = load_data_federated()

node1_train = idx_train1.send(node1)
node1_val = idx_val1.send(node1)
adj1 = adj_.send(node1)
features1 = features_.send(node1)
labels1 = labels_.send(node1)

node2_train = idx_train2.send(node2)
node2_val = idx_val2.send(node2)
adj2 = adj_.send(node2)
features2 = features_.send(node2)
labels2 = labels_.send(node2)

# Model and optimizer
model = GCN(nfeat=features_.shape[1],
            nhid=args.hidden,
            nclass=labels_.max().item() + 1,
            dropout=args.dropout)
# optimizer = optim.Adam(model.parameters(),
#                        lr=args.lr, weight_decay=args.weight_decay)

node1_model = model.copy().send(node1)
node2_model = model.copy().send(node2)
node1_optimizer = optim.Adam(node1_model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
node2_optimizer = optim.Adam(node2_model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features_ = features_.cuda()
    features1 = features1.cuda()
    features2 = features2.cuda()
    adj_ = adj_.cuda()
    labels_ = labels_.cuda()
    adj1 = adj1.cuda()
    labels1 = labels1.cuda()
    adj2 = adj2.cuda()
    labels2 = labels2.cuda()
    idx_train1 = idx_train1.cuda()
    idx_val1 = idx_val1.cuda()
    idx_train2 = idx_train2.cuda()
    idx_val2 = idx_val2.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    node1_model.train()
    node2_model.train()

    node1_optimizer.zero_grad()
    output1 = node1_model(features1, adj1)
    loss_train1 = F.nll_loss(output1[idx_train1], labels1[idx_train1])
    acc_train1 = accuracy(output1[idx_train1], labels1[idx_train1])
    loss_train1.backward()
    node1_optimizer.step()
    loss_train1 = loss_train1.get().data  # 注意这里获取标量值的方法

    node2_optimizer.zero_grad()
    output2 = node2_model(features2, adj2)
    loss_train2 = F.nll_loss(output2[idx_train2], labels2[idx_train2])
    acc_train2 = accuracy(output2[idx_train2], labels2[idx_train2])
    loss_train2.backward()
    node2_optimizer.step()
    loss_train2 = loss_train2.get().data  # 注意这里获取标量值的方法

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        node1_model.eval()
        node2_model.eval()
        output1 = node1_model(features1, adj1)
        output2 = node2_model(features2, adj2)

    loss_val1 = F.nll_loss(output1[idx_val1], labels1[idx_val1])
    acc_val1 = accuracy(output1[idx_val1], labels1[idx_val1])

    loss_val2 = F.nll_loss(output2[idx_val2], labels2[idx_val2])
    acc_val2 = accuracy(output2[idx_val2], labels2[idx_val2])

    # 模型移动到安全结点上计算均值
    node1_model.move(secure_worker)
    node2_model.move(secure_worker)

    with torch.no_grad():
        model.weight.set_(((node1_model.weight.data + node2_model.weight.data) / 2).get())
        model.bias.set_(((node1_model.bias.data + node2_model.bias.data) / 2).get())

    output = model(features_, adj_)
    loss_val = F.nll_loss(output[idx_val1], labels_[idx_val1])
    acc_val = accuracy(output[idx_val1], labels_[idx_val1])

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'loss_train1: {:.4f}'.format(loss_train1.item()),
          'acc_train1: {:.4f}'.format(acc_train1.item()),
          'loss_val1: {:.4f}'.format(loss_val1.item()),
          'acc_val1: {:.4f}'.format(acc_val1.item()),
          'loss_train2: {:.4f}'.format(loss_train2.item()),
          'acc_train2: {:.4f}'.format(acc_train2.item()),
          'loss_val2: {:.4f}'.format(loss_val2.item()),
          'acc_val2: {:.4f}'.format(acc_val2.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features_, adj_)
    loss_test = F.nll_loss(output[idx_test], labels_[idx_test])
    acc_test = accuracy(output[idx_test], labels_[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()

# 运行失败，如何分割图数据集是个问题
# 参考专栏：https://zhuanlan.zhihu.com/p/92602897
