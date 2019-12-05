# 前向传播描述网络结构
import tensorflow as tf

# 定义网络结构相关参数
INPUT_NODE = 784  # 神经网络输入节点784个，这784个点组成了一维数组
OUTPUT_NODE = 10  # 每个数表示对应的索引号出现的概率，实现了10分类
LAYER1_NODE = 500  # 定义了隐藏层的节点个数

def get_weight(shape,regularizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))  # 在训练神经网络是随机生成参数w
    if regularizer != None: # 如果使用正则化则将每个变量的正则化损失加入到总损失集合losses
        tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bais(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

# 搭建网络，描述从输入到输出的数据流
def forward(x,regularizer):
    w1 = get_weight([INPUT_NODE,LAYER1_NODE],regularizer) # 第一层参数w1
    b1 = get_bais([LAYER1_NODE]) # 第一层偏置b1
    y1 = tf.nn.relu(tf.matmul(x,w1)) + b1 # 第一层输出y1

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)  # 第二层参数w1
    b2 = get_bais([OUTPUT_NODE])  # 第二层偏置b1
    y = tf.matmul(y1, w2) + b2  # 第二层输出结果y直接输出，因为要对输出使之softmax函数使之符合概率分布
                                # 故输出y不过relu函数
    return y
