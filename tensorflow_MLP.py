import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.examples.tutorials.mnist import input_data

# 读取数据
mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)

in_units = 784  # 输入层节点数量
h1_units = 300  # 隐含层节点数量

x = tf.placeholder(tf.float32, [None, in_units])    # 定义输入x的placeholder
keep_prob = tf.placeholder(tf.float32)  # 定义dropout的比率

# 定义权重以及偏置
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

# 定义网络结构
hidden1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))  # 激活函数使用ReLU
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)    # 在隐含层使用Dropout
y = tf.nn.softmax(tf.add(tf.matmul(hidden1, W2), b2))   # 使用softmax分类

# 定义损失函数以及确定使用损失函数最小化的优化算法Adagrad
y_ = tf.placeholder(tf.float32, [None, 10])
cross_enerty = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), \
                              reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_enerty)

# 定义会话以及初始化全部变量
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 共使用3000个batch，每个batch100个样本进行训练模型
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

# 计算模型的精度
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 对测试集进行精度计算
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, \
                     keep_prob: 1.0}))
