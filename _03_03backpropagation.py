#coding = utf-8
#0导入模块，生成模拟数据集
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
seed = 23455

#基于seed产生随机数
rng = np.random.RandomState(seed)
#随机数返回32行2列（32组数据 重量、体积）作为输入输入集
X = rng.rand(32,2)
#从32行2列的的矩阵中 取出一行 判断如果和小于1 给Y赋值1 如果和不小于1 给Y赋值0
#作为输入数据集的标签（零件是否合格）
Y = [[int(x0 + x1 < 1)] for [x0,x1] in X]
print('X\n',X)
print('Y\n',Y)

#1定义神经网络的输入、参数和输出，定义前向传播过程；
x = tf.placeholder(tf.float32,shape=(None,2)) #None表示输入的数据组数（维数）未知，每组的特征有2个
y_ = tf.placeholder(tf.float32,shape=(None,1)) #有1个输出

w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1)) #初始的参数设置为随机的正态分布
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

a = tf.matmul(x,w1) #矩阵乘法 输入的数据与w1参数相乘的结果为a，输入层的神经元数为1，隐含层的神经元数为3，输出层的神经元数为1
y = tf.matmul(a,w2) #a与w2参数矩阵相乘

#2定义损失函数及反向传播方法
loss = tf.reduce_mean(tf.square(y-y_)) #mean取平均值，square去平方，均方误差，y为预测值、y_为已知答案
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss) #0.001为学习率,优化方法为梯度下降
#train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
#train_step = tf.train.AdadeltaOptimizer(0.001).minimize(loss)

#3生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op) #将初始化参数传入会话
    #输出目前（未经训练）的参数取值
    print('w1\n',sess.run(w1))
    print('w2\n',sess.run(w2))
    print('\n')

    #训练模型
    STEPS = 3000 #迭代次数为3000次
    for i in range(STEPS): #每轮从X、Y中抽取对应的的从start开始到end结束个特征值和标签，喂入神经网络
        start = (i*BATCH_SIZE)%32
        end = start + BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i % 500 == 0: #每500轮打印一次loss值
            total_loss = sess.run(loss,feed_dict={x:X,y_:Y})
            print('After %d training step(s),loss on all data is %g'%(i,total_loss))
    #输出训练后的参数取值
    print('\n')
    print('w1\n',sess.run(w1)) #经过3000轮打印出最终训练好的参数w1、w2
    print('w2\n',sess.run(w2))
    print('我好气啊，11点半就打完代码，找了一个小时的bug,在想为什么feed_dict中的键不能传递给张量，原因是脑子坏掉了中途加了一个y_ = 1,'
          '还有开始，定义输入时将y_写成y,真的是眼瞎、手欠')
