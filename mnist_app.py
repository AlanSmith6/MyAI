# coding = utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_backward
import mnist_forward


def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:  # 重现计算图
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])  # 仅需要给x占位
        y = mnist_forward.forward(x, None)  # 计算求得输出y
        preValue = tf.argmax(y, 1)  # y的最大值对应的列表索引号就是预测结果

        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)  # 实例化带有滑动平均值的saver

        with tf.Session() as sess:  # 用with结构加载ckpt
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:  # 如果ckpt存在则恢复ckpt参数的相关信息到当前会话
                saver.restore(sess,ckpt.model_checkpoint_path)

                preValue = sess.run(preValue,feed_dict={x:testPicArr})  # 将准备好的图片喂入神经网络，执行预测操作
                return preValue
            else:
                print('No checkpoint file found')  # 若没有找到ckpt给出提示
                return -1
# 完成输入图片的预处理操作，符合神经网络对输入特征的要求
def pre_pic(picName):
    img = Image.open(picName)  # 先打开传入的原始图片
    reIm = img.resize((28,28),Image.ANTIALIAS)  # 为符合模型对图片尺寸的要求：将原始图片重制为28*28size的图片，NATIALIAS表示用消除锯齿的方式resize
    im_arr = np.array(reIm.convert('L'))  # 为符合模型对图片颜色的要求，convert将图片变为灰度图，array将图像变为矩阵的形式
    threshold = 50
    for i in range(28):  # 模型要求输入为黑底白字，而我们输入的为白底黑字，要给输入图片反色
        for j in range(28):  # 利用两个循环遍历所有像素点
            im_arr[i][j] = 255 - im_arr[i][j]  # 每个像素点新值
            if (im_arr[i][j]) < threshold:  # （threshold为阈值）给图片做二值化处理，让图片只有纯白色和纯黑色点
                im_arr[i][j] = 0            # 这样可以滤掉输入图片的噪声，留下图像主要特征
            else:
                im_arr[i][j] = 255
    nm_arr = im_arr.reshape([1,784])  # 整理形状为一行784列
    nm_arr = nm_arr.astype(np.float32)  # 按要求将像素值改为0到1之间的浮点数
    img_ready = np.multiply(nm_arr,1.0/255.0)  # 再将现有的RGB图从0到255之间的数变为0到1之间的浮点数 数

    return img_ready

def application():
    testNum = int(input('input the number of test pictures:'))
    for i in range(testNum):
        testPic = input('the path of test picture:')  # 给出要识别文件的路径和名称
        testPicArr = pre_pic(testPic)  # 先把接收到的图片交给Pic函数做预处理
        preValue = restore_model(testPicArr)  # 将待识别的图片喂入神经网络
        print('The prediction number is :',preValue)

def main():  # 在main函数中调用了application函数
    application()
if __name__ == '__main__':
    main()