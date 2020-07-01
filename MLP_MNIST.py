from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np

# 设定随机数种子
seed = 7
np.random.seed(seed)

# 导入MNIST数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 显示四张图
# plt.subplot(221)
# plt.imshow(X_train[0], cmap=plt.get_cmap())
#
# plt.subplot(222)
# plt.imshow(X_train[1], cmap=plt.get_cmap())
#
# plt.subplot(223)
# plt.imshow(X_train[2], cmap=plt.get_cmap())
#
# plt.subplot(224)
# plt.imshow(X_train[3], cmap=plt.get_cmap())

# plt.show()

num_pixels = X_train.shape[1] * X_train.shape[2]
print(X_train.shape[1])
print(X_train.shape[2])
print(num_pixels)
print(X_train.shape[0])
print(X_test.shape[0])

x_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')  # 转换为1维向量
x_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# 格式化数据到0-1
x_train_normalize = x_train / 255  # 标准化
x_test_normalize = x_test / 255

# 进行one-hot编码
y_trainOneHot = np_utils.to_categorical(y_train)
y_testOneHot = np_utils.to_categorical(y_test)
num_class = y_testOneHot.shape[1]
print(num_class)


# 建立模型
def create_model():
    # 创建模型
    model = Sequential()
    model.add(Dense(units=num_pixels, input_dim=num_pixels,
                    kernel_initializer='normal', activation='relu'))  # 输入层隐藏层
    model.add(Dense(units=num_class, input_dim=num_pixels,
                    kernel_initializer='normal', activation='softmax'))  # 输出层
    model.summary()  # 查看模型摘要

    # 训练模型
    # 设置训练方式，loss为设置损失函数categorical_crossentropy（交叉熵）的训练效果比较好，
    # 一般使用这个（损失函数就是帮我我们计算真实值和预测值之间的误差）。optimizer为设置优
    # 化器，使用使用adam优化器可以让模型更快收敛（优化器的作用为在不断的批次训练中不断更
    # 新权重和偏差，是损失函数最小化）。metrics为设置评估模型的方式为准确率。
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])  # 定义训练方式

    return model


# 评估模型准确率
model = create_model()
# 开始训练的参数里面输入训练数据，训练标签，划分0.2为验证集，epochs为训练周期，
# batch_size为每一训练批次要输入多少个数据（训练批次 = 总数据 / 一批次训练的数据
# 量），verbose为显示训练过程。
train_history = model.fit(x=x_train_normalize,
                          y=y_trainOneHot, validation_split=0.2,
                          epochs=10, batch_size=200, verbose=2)  # 设置训练参数

scores = model.evaluate(x_test_normalize, y_testOneHot)  # 评估测试集
print('MLP的准确率为: %.2f%%' % (scores[1] * 100))


# 建立显示训练过程的函数
# def show_train_history(train_history, train, validation):
#     plt.plot(train_history.history[train])
#     plt.plot(train_history.history[validation])
#     plt.title('Train History')
#     plt.ylabel(train)
#     plt.xlabel('Epoch')
#     plt.legend(['train', 'validation'], loc='upper left')
#     plt.show()
#
#
# show_train_history(train_history, 'acc', 'val_acc')  # 画出准确率执行结果
prediction=model.predict_classes(x_test)  # 预测测试集


def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()  # 设置显示图形大小
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1+i)  # 建立子图5行5列
        ax.imshow(images[idx], cmap='binary')
        title = 'lablel='+str(labels[idx])
        if len(prediction) > 0:
            title += ',predict=' + str(prediction[idx])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        idx += 1
    plt.show()


plot_images_labels_prediction(X_test, y_test, prediction, idx=340)