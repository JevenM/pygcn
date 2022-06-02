import torch as th
import syft as sy
hook = sy.TorchHook(th)
from torch import nn, optim

bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")

secure_worker = sy.VirtualWorker(hook, id="secure_worker")


# A Toy Dataset
data = th.tensor([[0,0],[0,1],[1,0],[1,1.]], requires_grad=True)
target = th.tensor([[0],[0],[1],[1.]], requires_grad=True)

# get pointers to training data on each worker by
# sending some training data to bob and alice
bobs_data = data[0:2].send(bob)
bobs_target = target[0:2].send(bob)

alices_data = data[2:].send(alice)
alices_target = target[2:].send(alice)


model = nn.Linear(2,1)

iterations = 10
worker_iters = 5

# 外层循环控制全局模型的更新轮次
for a_iter in range(iterations):

    # 复制模型并拷贝到多个结点
    bobs_model = model.copy().send(bob)
    alices_model = model.copy().send(alice)

    bobs_opt = optim.SGD(params=bobs_model.parameters(), lr=0.1)
    alices_opt = optim.SGD(params=alices_model.parameters(), lr=0.1)

    # 在各自结点上训练几个轮次
    for wi in range(worker_iters):
        # Train Bob's Model
        bobs_opt.zero_grad()
        bobs_pred = bobs_model(bobs_data)
        bobs_loss = ((bobs_pred - bobs_target) ** 2).sum()
        bobs_loss.backward()

        bobs_opt.step()
        bobs_loss = bobs_loss.get().data

        # Train Alice's Model
        alices_opt.zero_grad()
        alices_pred = alices_model(alices_data)
        alices_loss = ((alices_pred - alices_target) ** 2).sum()
        alices_loss.backward()

        alices_opt.step()
        alices_loss = alices_loss.get().data
    # 模型移动到安全结点上计算均值
    alices_model.move(secure_worker)
    bobs_model.move(secure_worker)

    with th.no_grad():
        model.weight.set_(((alices_model.weight.data + bobs_model.weight.data) / 2).get())
        model.bias.set_(((alices_model.bias.data + bobs_model.bias.data) / 2).get())

    print("Bob:" + str(bobs_loss) + " Alice:" + str(alices_loss))
    preds = model(data)
    loss = ((preds - target) ** 2).sum()
    # print(loss.data)
    print(loss.data.item())








#
#
# n_features = boston_data['alice'][0].shape[1]
# n_targets = 1
#
# model = th.nn.Linear(n_features, n_targets)
#
# # Cast the result in BaseDatasets
# datasets = []
# for worker in boston_data.keys():
#     dataset = sy.BaseDataset(boston_data[worker][0], boston_target[worker][0])
#     datasets.append(dataset)
#
# # Build the FederatedDataset object
# dataset = sy.FederatedDataset(datasets)
# print(dataset.workers)
# optimizers = {}
# for worker in dataset.workers:
#     optimizers[worker] = th.optim.Adam(params=model.parameters(),lr=1e-2)
# # ['bob', 'theo', 'jason', 'alice', 'andy', 'jon']
#
# train_loader = sy.FederatedDataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)
#
# epochs = 50
# for epoch in range(1, epochs + 1):
#     loss_accum = 0
#     for batch_idx, (data, target) in enumerate(train_loader):
#         model.send(data.location)
#
#         optimizer = optimizers[data.location.id]
#         optimizer.zero_grad()
#         pred = model(data)
#         loss = ((pred.view(-1) - target) ** 2).mean()
#         loss.backward()
#         optimizer.step()
#
#         model.get()
#         loss = loss.get()
#
#         loss_accum += float(loss)
#
#         if batch_idx % 8 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch loss: {:.6f}'.format(
#                 epoch, batch_idx, len(train_loader),
#                 100. * batch_idx / len(train_loader), loss.item()))
#
#     print('Total loss', loss_accum)