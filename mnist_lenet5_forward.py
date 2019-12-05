# coding = utf-8
import tensorflow as tf

# 搭建神经网络所用到的常数
IMAGE_SIZE = 28  # mnist数据集每张图片的分辨率，也就是横向和纵向的边长
NUM_CHANNELS = 1  # 是灰度图，所以通道数为1
CONV1_SIZE = 5  # 第一层卷积核的大小是5
CONV1_KERNEL_NUM = 32  # 第一层使用了32个卷积核
CONV2_SIZE = 5  # 第二层卷积核的大小是5
CONV2_KERNEL_NUM = 64  # 第二层使用了64个卷积核
FC_SIZE = 512  # 第一层神经网络有512个神经元
OUTPUT_NODE = 10  # 第二层神经元有10个神经元，对应了10分类的输出

def get_weight(shape,regularizer):  # 参数分别为生成参数的维度，以及正则化
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))  # 生成去掉过大偏离点的正态分布随机数，返回随机初始化的参数w
    if regularizer != None:tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w));
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))  # 生成初始量为0的偏置b
    return b

# 卷积计算函数
def conv2d(x,w):  # 输入图片x和所用卷积核w
    # x为输入描述eg:[batch(一次喂入图片的数量)，5(行分辨率),5(列分辨率),3(输入通道数)]，w为卷积核描述eg:[3(行分辨率),3(列分辨率),3(通道数),16(核个数)]
    return tf.nn.conv2d(x,w,strides=[1, 1, 1, 1],padding='SAME')  # strides为核滑动步长中间两个1表示横向纵向滑动都为1,使用0填充

# 最大池化计算函数
def max_pool_2x2(x):
    # x为输入描述eg:[batch,28,28,6],ksize为池化核描述eg:[1,2(行长),2(列长),1]，strides为核滑动步长描述
    return tf.nn.max_pool(x,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')

# forward函数给出了前向传播的网络结构
def forward(x,train,regularizer):
    # 初始化第一层卷积核依次为，行分辨率、列分辨率、通道数、核的个数
    conv1_w = get_weight([CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_KERNEL_NUM], regularizer)
    # 初始化第一层偏置
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    # 执行卷积计算，输入是x,卷积核是初始化的conv1_w
    conv1 = conv2d(x,conv1_w)
    # 为卷积后的conv1添加偏置，通过激活函数
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_b))
    # 将激活后的输出进行最大池化
    pool1 = max_pool_2x2(relu1)

    # 第二层卷积核的深度等于第一层卷积核的个数
    conv2_w = get_weight([CONV2_SIZE,CONV2_SIZE,CONV1_KERNEL_NUM,CONV2_KERNEL_NUM],regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    # 这层卷积层的输入时上一层的输出pool1
    conv2 = conv2d(pool1,conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_b))
    pool2 = max_pool_2x2(relu2)  # pool2是第二层卷积的输出，需要把它从三维张量变为二维张量

    pool_shape = pool2.get_shape().as_list() # 得到pool2输出矩阵的维度存入list中
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]  # 提取特征的长度、宽度、深度，三者相乘就是可以得到特征点的个数
    # 将pool2表示成batch行，所有特征点作为个数列的二维形状，再把它喂到全连接网络里
    reshaped = tf.reshape(pool2,[pool_shape[0],nodes])  # pool_shape[0]是一个batch的值

    fc1_w = get_weight([nodes,FC_SIZE],regularizer)
    fc1_b = get_bias([FC_SIZE])
    # 把上层的输出乘以本层现场的权重加上偏置过激活函数relu
    fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_w) + fc1_b)
    # 如果是训练阶段，则对该层的输出使用50%的dropout
    if train: fc1 = tf.nn.dropout(fc1,0.5)

    # 通过第二层全连接网络，初始化w,b
    fc2_w = get_weight([FC_SIZE,OUTPUT_NODE],regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    # 上层的输入乘以本层的权重加上本层的偏置得到输出y
    y = tf.matmul(fc1,fc2_w) + fc2_b
    return y