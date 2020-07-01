from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np

# 导入MNIST数据
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 显示四张图
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap())

plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap())

plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap())

plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap())

# plt.show()

num_pixels = X_train.shape[1]*X_train.shape[2]
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
num_class = y_test.shape[1]
print(num_class)

# 建立模型
def create_model():
    model = Sequential()
    model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu'))