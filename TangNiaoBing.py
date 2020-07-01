from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#设定随机数种子
np.random.seed(7)

#导入数据
dataset = np.loadtxt(r'F:\data\PimaIndiansdiabetes.csv', delimiter=',', skiprows=1)
# dataset = np.loadtxt('F:\PimaIndiansdiabetes.csv', dalimiter=',')

x = dataset[:, 0:8]
y = dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#编译模型——二分类中损失函数定义为“二进制交叉熵”，优化器Adam为有效的梯度下降算法
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

#训练模型——epochs参数表示对数据集进行固定次数的迭代
#batch_size表示在执行神经网络中的权重更新的每个批次中所用实例的个数
model.fit(x=x, y=y, epochs=150, batch_size=10)

#评估模型——这里简化用，训练集评估
scores = model.evaluate(x=x, y=y)
print('\n%s: %.2f%%' %(model.metrics_names[1], scores[1]*100))
