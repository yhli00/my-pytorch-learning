import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
import os

train_path = 'mnist/mnist_train.csv'
test_path = 'mnist/mnist_test.csv'
save_dir = 'model_save/'


def loadData(filename):
    datas = []
    labels = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            labels.append(int(line[0]))
            datas.append([int(i) / 255 for i in line[1:]])
    return datas, labels

class mnist_data(Dataset):
    def __init__(self, datas, labels):
        datas = np.array(datas)
        labels = np.array(labels)
        self.x = torch.from_numpy(datas).float().to('cuda')
        self.y = torch.from_numpy(labels).to('cuda')
        self.len = len(datas)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)

def one_hot(labels):
    labels = labels.view(-1, 1)
    tmp = torch.zeros(labels.shape[0], 10).to('cuda')
    return tmp.scatter_(1, labels, 1)


def train(epochs, data_loader, net, optimizer, print_every=50):
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(data_loader):
            out = net(x)
            loss = F.mse_loss(out, one_hot(y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % print_every == 0:
                print(f'epoch:[{epoch + 1}/{epochs}] [{batch_idx}/{len(data_loader)}] loss={loss.item()}', flush=True)
    return epochs, loss

def test(test_loader, net):
    correct = 0
    total = 0
    for x, y in test_loader:
        out = net(x)
        _, out = out.max(dim=1)
        correct += out.eq(y).sum().float()
        total += len(y)
    return correct / total

# 保存模型，单卡训练和多卡训练都一样
def save_model(epoch, loss, model, save_dir, filename):
    checkpoint = {}
    checkpoint['epoch'] = epoch
    checkpoint['loss'] = loss.item()
    checkpoint['state_dict'] = model.state_dict()
    torch.save(checkpoint, os.path.join(save_dir, filename))

# 从单卡训练好的模型文件到单卡gpu上 
def load_from_single_gpu_trained(save_dir, filename, device='cuda'):
    
    # 加载到gpu上
    if device == 'cuda':
        checkpoint = torch.load(os.path.join(save_dir, filename))
        net = Net().to('cuda')
        net.load_state_dict(checkpoint['state_dict'])
    # 加载到cpu上
    else:
        checkpoint = torch.load(os.path.join(save_dir, filename), map_location=torch.device('cpu'))
        net = Net()
        net.load_state_dict(checkpoint['state_dict'])
    return net, checkpoint['epoch'], checkpoint['loss']

# 单卡训练，单卡加载模型文件
def single_train_single_load(train_data, test_data):
    device = torch.device('cuda')
    net = Net()
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    epoch, loss = train(20, train_data, net, optimizer)
    save_model(epoch, loss, net, save_dir, 'model.mdl')
    load_model, epoch, loss = load_from_single_gpu_trained(save_dir, 'model.mdl')
    acc = test(test_data, load_model)
    print(f'acc={acc}, epoch={epoch} loss={loss}')


# 从多卡训练的模型文件加载到多卡上
def load_from_multi_gpu(save_dir, filename, to_multi_gpu=True):
    import os
    checkpoint = torch.load(os.path.join(save_dir, filename))
    if to_multi_gpu:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'  # 指定使用第0个gpu和第一个gpu
        net = Net()
        net = nn.DataParallel(net).to('cuda')
        net.load_state_dict(checkpoint['state_dict'])
    else:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        net = Net().to('cuda')
        net = nn.DataParallel(net)
        net.load_state_dict(checkpoint['state_dict'])  # 单卡加载

    return net, checkpoint['epoch'], checkpoint['loss']


# 多卡训练，多卡/单卡加载模型文件
def muti_train_multi_load(train_data, test_data):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'  # 指定使用第0个gpu和第一个gpu
    net = Net()
    net = nn.DataParallel(net).to('cuda')
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    epoch, loss = train(10, train_data, net, optimizer)
    save_model(epoch, loss, net, save_dir, 'model.mdl')
    # load_model, epoch, loss = load_from_multi_gpu(save_dir, 'model.mdl')  # 多卡加载文件
    load_model, epoch, loss = load_from_multi_gpu(save_dir, 'model.mdl', to_multi_gpu=False)  # 单卡加载文件
    acc = test(test_data, load_model)
    print(f'acc={acc}, epoch={epoch} loss={loss}')

# 加载多卡训练好的文件到cpu
def load_from_multi_gpu_to_cpu(save_dir, filename):
    checkpoint = torch.load(os.path.join(save_dir, filename))
    net = Net().to('cpu')
    net = nn.DataParallel(net)
    net.load_state_dict(checkpoint['state_dict'])


if __name__ == '__main__':
    train_data = DataLoader(mnist_data(*loadData(train_path)), batch_size=512, shuffle=True)
    test_data = DataLoader(mnist_data(*loadData(test_path)), batch_size=512, shuffle=True)
    # single_train_single_load(train_data, test_data)
    # muti_train_multi_load(train_data, test_data)
    # load_from_multi_gpu_to_cpu(save_dir, 'model.mdl')
    load_model, epoch, loss = load_from_multi_gpu(save_dir, 'model.mdl', to_multi_gpu=False)  # 单卡加载文件
    acc = test(test_data, load_model)
    print(f'acc={acc}, epoch={epoch} loss={loss}')
    

