# coding:utf-8
# 0导入模块，生成模拟数据集
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import _04_05generateds
import _04_05forward

STEPS = 40000
BATCH_SIZE = 30
LEARNING_RATE_BASE = 0.001 # 学习率基数
LEARNING_RATE_DECAY = 0.999 # 学习率衰减率
REGULARIZER = 0.01

def backward():
    x = tf.placeholder(tf.float32,shape=(None,2))
    y_ = tf.placeholder(tf.float32,shape=(None,1))

    X,Y_,Y_c = _04_05generateds.generateds()
    y = _04_05forward.forward(x,REGULARIZER)
    global_step = tf.Variable(0,trainable=False)
    # 指数衰减学习率
    learning_step = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        300/BATCH_SIZE, # 多久更新一次学习率
        LEARNING_RATE_DECAY,
        staircase=True)

    # 定义损失函数
    loss_mse = tf.reduce_mean(tf.square(y-y_))
    loss_total = loss_mse + tf.add_n(tf.get_collection('losses')) #添加正则化

    # 定义反向传播方法：包含正则化
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEPS):
            start = (i*BATCH_SIZE) %300
            end = start +BATCH_SIZE
            sess.run(train_step,feed_dict={x: X[start:end],y_:Y_[start:end]})
            if i%2000 == 0:
                loss_v = sess.run(loss_total,feed_dict={x:X,y_:Y_})
                print("After %d steps,loss is :%f"%(i,loss_v))

        xx,yy = np.mgrid[-3:3:0.01,-3:3:0.01] # 生成以0.01为分辨率的网格坐标点
        # np.c_ 将xx坐标和yy坐标点对应位置配对成矩阵，组成网格坐标点
        grid = np.c_[xx.ravel(),yy.ravel()] # ravel将xx坐标拉直变为一行n列
        probs = sess.run(y,feed_dict={x:grid})
        probs = probs.reshape(xx.shape)
    plt.scatter(X[:,0],X[:,1],c = np.squeeze(Y_c))
    plt.contour(xx,yy,probs,levels=[.5]) # level为等高线的高度
    plt.show()

if __name__ =='__main__':  # 只有在执行本文件时才会执行语段，其他文件调用该文件时则不会执行本语段
    backward()