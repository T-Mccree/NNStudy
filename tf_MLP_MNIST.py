import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import numpy as np


mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)           #下载据数
print('train images:',mnist.train.images.shape,                       #查看数据
     'labels:',mnist.train.labels.shape)
print('validation images:',mnist.validation.images.shape,
     'labels:',mnist.validation.labels.shape)
print('test images:',mnist.test.images.shape,
     'labels:',mnist.test.labels.shape)
#定义显示多项图像的函数
def plot_images_labels_prediction_3(images,labels,prediction,idx,num=10):
    fig=plt.gcf()
    fig.set_size_inches(12,14)
    if num>25:num=25
    for i in range(0,num):
        ax=plt.subplot(5,5,i+1)
        ax.imshow(np.reshape(images[idx],(28,28)),cmap='binary')
        title='lable='+str(np.argmax(labels[idx]))
        if len(prediction)>0:
            title+=",prediction="+str(prediction[idx])
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()

plot_images_labels_prediction_3(mnist.train.images,mnist.train.labels,[],0)
#定义layer函数，构建多层感知器模型
def layer(output_dim,input_dim,inputs,activation=None):
    W=tf.Variable(tf.random_normal([input_dim,output_dim]))
    b=tf.Variable(tf.random_normal([1,output_dim]))
    XWb=tf.matmul(inputs,W)+b
    if activation is None:
        outputs=XWb
    else:
        outputs=activation(XWb)
    return outputs
#建立输入层
x=tf.placeholder("float",[None,784])
#建立隐藏层
h1=layer(output_dim=256,input_dim=784,inputs=x,
       activation=tf.nn.relu)
#建立输出层
y_predict=layer(output_dim=10,input_dim=256,inputs=h1,
        activation=None)
y_label=tf.placeholder("float",[None,10])
#定义损失函数
loss_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                             (logits=y_predict,
                              labels=y_label))
#定义优化器
optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)
#计算每一项数据是否预测正确
correct_prediction=tf.equal(tf.argmax(y_label,1),
                            tf.argmax(y_predict,1))
#计算预测正确结果的平均值
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
#1、定义训练参数
trainEpochs=15                                   #设置执行15个训练周期
batchSize=100                                    #每一批次项数为100
totalBatchs=int(mnist.train.num_examples/batchSize)        #计算每个训练周期
loss_list=[];epoch_list=[];accuracy_list=[]      #初始化训练周期、误差、准确率
from time import time                            #导入时间模块
startTime=time()                                 #开始计算时间
sess=tf.Session()                                #建立Session
sess.run(tf.global_variables_initializer())       #初始化TensorFlow global 变量
#2、进行训练
for epoch in range(trainEpochs):
    for i in range(totalBatchs):
        batch_x,batch_y=mnist.train.next_batch(batchSize)  #使用mnist.train.next_batch方法读取批次数据，传入参数batchSize是100
        sess.run(optimizer,feed_dict={x:batch_x,
                                      y_label:batch_y})    #执行批次训练
    loss,acc=sess.run([loss_function,accuracy],            #使用验证数据计算准确率
                      feed_dict={x:mnist.validation.images,
                                 y_label:mnist.validation.labels})
    epoch_list.append(epoch);                              #加入训练周期列表
    loss_list.append(loss)                                 #加入误差列表
    accuracy_list.append(acc)                              #加入准确率列表
    print("Train Epoch:",'%02d' % (epoch+1),"Loss=",\
          "{:.9f}".format(loss),"Accuracy=",acc)
duration=time()-startTime
print("Train Finished takes:",duration)                    #计算并显示全部训练所需时间
#画出误差执行结果

fig=plt.gcf()
fig.set_size_inches(4,2)
plt.plot(epoch_list,loss_list,label='loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'],loc='upper left')
#画出准确率执行结果
plt.plot(epoch_list,accuracy_list,label="accuracy")
fig=plt.gcf()
fig.set_size_inches(4,2)
plt.ylim(0.8,1)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
#评估模型准确率
print("accuracy:",sess.run(accuracy,
                           feed_dict={x:mnist.test.images,
                                      y_label:mnist.test.labels}))
#进行预测
#1.执行预测
prediction_result=sess.run(tf.argmax(y_predict,1),
                           feed_dict={x:mnist.test.images})
#2.预测结果
print(prediction_result[:10])
#3.显示前10项预测结果
plot_images_labels_prediction_3(mnist.test.images,
                              mnist.test.labels,
                              prediction_result,0)