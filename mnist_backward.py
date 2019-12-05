# coding = utf-8
# 反向传播描述了参数的优化方法
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os

INPUT_NODE = 784  # 神经网络输入节点784个，这784个点组成了一维数组
OUTPUT_NODE = 10  # 每个数表示对应的索引号出现的概率，实现了10分类
BATCH_SIZE = 200  # 定义每轮喂入神经网络多少张图片
LEARNING_RATE_BASE = 0.1  # 最开始的学习率
LEARNING_RATE_DECAY = 0.99  #学习率衰减率
REGULARIZER = 0.0001  # 正则化系数
STEPS = 5000  # 共训练多少轮
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率
MODEL_SAVE_PATH = 'F:\PyCharm\myAI\model'  # 模型保存路径
MODEL_NAME = 'mnist_model'  # 模型保存文件名

# 在backward函数中读入mnist
def backward(mnist):
    # 首先利用placeholder给x和y_占位
    x = tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32,[None,mnist_forward.OUTPUT_NODE])
    y = mnist_forward.forward(x,REGULARIZER)  # 调用前向传播的程序计算输出y
    global_step = tf.Variable(0,trainable=False)  # 给轮数计数器赋初值并设置为不可训练

    # 调用包含正则化的损失函数losses
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    # 定义指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase = True)

    # 定义训练过程
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    #定义滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op = tf.no_op(name='train')

    # 实例化saver
    saver = tf.train.Saver()

    # 在with结构中初始化所有变量
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        #ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        ckpt = tf.train.get_checkpoint_state('F:\PyCharm\myAI\model')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpont_path)

        #用for循环迭代steps轮
        for i in range(STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)  # 每次读入batch_size组数据和标签
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y:ys})  # 把他们喂入神经网络，
                                                                                             # 执行训练过程
            if i % 1000 == 0:  # 每一千轮打印出当前的loss值，要在sess.run运行后才会有结果
                print('After %d training steps(s)，loss_in_training batch is %g.'%(step,loss_value))
                # 保存模型和当前会话
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main():
    mnist = input_data.read_data_sets('./data/',one_hot=True)
    backward(mnist)

if __name__ == '__main__':  # 只有在执行本文件时才会执行语段，其他文件调用该文件时则不会执行本语段
    main()