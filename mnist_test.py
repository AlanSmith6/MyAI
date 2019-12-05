# utf-8
# 测试复现了节点 计算模型在测试集中的准确率

import  time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
TEST_INTERVAL_SECS = 5  # 定义程序循环间隔时间是5s

def test(mnist):  # 将mnist数据集读入test函数
    with tf.Graph().as_default() as g:  # 复现计算图
        x = tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32,[None,mnist_forward.OUTPUT_NODE])
        y = mnist_forward.forward(x,None)  # 利用前向传播过程计算出y的值

        # 实例化带滑动平均的saver对象，这样所有参数在会话中被加载时会被赋值为各自的滑动平均值
        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        # 计算准确率
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        while True:
            with tf.Session() as sess:  # 在with结构中加载ckpt，将滑动平均值赋给各个参数
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:  # 判断是否存在模型
                    saver.restore(sess,ckpt.model_checkponint_path)  # 若存在恢复模型到当前会话
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]  # 恢复global值
                    accuracy_score = sess.run(accuracy,feed_dict={x:mnist.test.image,y_:mnist.test.labels})   # 执行准确率计算
                    print('After %s training step(s),test accuracy = %g' % (global_step,accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)

# 读入数据集调用test函数
def main():
    mnist = input_data.read_data_sets('./data/',one_hot = True)
    test(mnist)

if __name__ == '__main__':
    main()
