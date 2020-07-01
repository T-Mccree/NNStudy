from keras.datasets import mnist
import matplotlib.pyplot as plt

#导入MNIST数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#显示四张图
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap())

plt.show()
