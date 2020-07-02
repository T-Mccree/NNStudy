import torch.nn as nn
import torch
import torchvision
from torch.utils.data import DataLoader
import time

print("start")
EPOCH = 50  # 总的训练次数
BATCH_SIZE = 20  # 批次的大小
LR = 0.03  # 学习率#交叉熵损失函数不需要太大的学习率
DOWNLOAD_MNIST = False  # 运行代码的时候是否下载数据集

cuda_available = torch.cuda.is_available()  # 获取GPU是否可用，可用的话就用GPU进行训练和测试
# #对于这样的网络，可能cpu更快一些
cuda_available = False  # 即使gpu可用，也可以执行这一句，测试训练在cpu上的训练速度
# 设置一个转换的集合，先把数据转换到tensor，再归一化为均值.5，标准差.5的正态分布
trans = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),  # ToTensor方法把[0,255]变成[0,1]
        torchvision.transforms.Normalize([0.5], [0.5])  # mean(均值)，std（标准差standard deviation）
    ]
)

print("load data")
train_data = torchvision.datasets.MNIST(
    root="./mnist",  # 设置数据集的根目录
    train=True,  # 是否是训练集
    transform=trans,  # 对数据进行转换
    download=DOWNLOAD_MNIST
)
# 第二个参数是数据分块之后每一个块的大小，第三个参数是是否大乱数据
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(
    root="./mnist",
    train=False,  # 测试集，所以false
    transform=trans,
    download=DOWNLOAD_MNIST
)
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
print("net creating")

# 用最简单的方式搭建一个dnn
net = torch.nn.Sequential(
    nn.Linear(28 * 28, 30),  # 输入28*28个，输出30个
    nn.Tanh(),  # 激活函数
    nn.Linear(30, 10)  # 输入30个，输出10个，对应10个类别
)

if cuda_available:
    net.cuda()  # 使用GPU

# 定义损失函数和优化方法
loss_function = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = torch.optim.SGD(net.parameters(), lr=LR)  # 优化方法

print("start training")
for ep in range(EPOCH):
    # 记录把所有数据集训练+测试一遍需要多长时间
    startTick = time.clock()
    for data in train_loader:  # 对于训练集的每一个batch
        img, label = data
        img = img.view(img.size(0), -1)  # 拉平图片成一维向量

        if cuda_available:
            img = img.cuda()
            label = label.cuda()

        out = net(img)  # 送进网络进行输出
        loss = loss_function(out, label)  # 获得损失

        optimizer.zero_grad()  # 梯度归零
        loss.backward()  # 反向传播获得梯度，但是参数还没有更新
        optimizer.step()  # 更新梯度

    num_correct = 0  # 正确分类的个数，在测试集中测试准确率
    # 由于测试集的batchsize是测试集的长度，所以下面的循环只有一遍
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)

        if cuda_available:
            img = img.cuda()
            label = label.cuda()

        out = net(img)  # 获得输出

        _, prediction = torch.max(out, 1)
        print(prediction)
        # torch.max()返回两个结果，
        # 第一个是最大值，第二个是对应的索引值；
        # 第二个参数 0 代表按列取最大值并返回对应的行索引值，1 代表按行取最大值并返回对应的列索引值。
        num_correct += (prediction == label).sum()  # 找出预测和真实值相同的数量，也就是以预测正确的数量

    accuracy = num_correct.cpu().numpy() / len(test_data)  # 计算正确率，num_correct是gpu上的变量，先转换成cpu变量
    timeSpan = time.clock() - startTick
    print("第%d迭代期，准确率为%f,耗时%dS" % (ep + 1, accuracy, timeSpan))
