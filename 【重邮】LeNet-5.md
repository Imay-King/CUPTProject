# 【重邮】LeNet-5-（初稿）

~~~
熟练掌握基于TensorFlow的深度卷积神经网络的原理、实现分类网络LeNet5，并在手写字符识别数据集MNIST上验证和评价LeNet5的分类性能。
~~~

### 1-数据集的介绍

机器学习（或深度学习）的入门教程，一般都是 [MNIST](http://yann.lecun.com/exdb/mnist/) 数据库上的手写识别问题。MNIST数据集作为一个简单的计算机视觉数据集，包含一系列如图1所示的手写数字图片和对应的标签。图片是28x28的像素矩阵，标签则对应着0~9的10个数字。每张图片都经过了大小归一化和居中处理

![](C:\Users\Administrator\Desktop\Lenet\图片\MNIST.png)

#### 1.1加载数据集

~~~python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)#加载图片并将标记进行ont-hot编码
~~~

>  下载下来的数据集被分为三部分
>
>  | 数据             | 数据量 |
>  | ---------------- | ------ |
>  | mnist.train      | 55000  |
>  | mnist.test       | 10000  |
>  | mnist.validation | 5000   |
>
>  MNIST数据集的训练数据集（`mnist.train.images`）是一个 55000 * 784 的矩阵，矩阵的每一行代表一张图片（28 * 28 * 1）的数据，图片的数据范围是 [0, 1]，代表像素点灰度归一化后的值。

### 2-算法原理介绍

#### 2.1卷积神经网络

LeNet-5是一个比较简单的卷积神经网络，下图显示了其结构，先经过两次卷积层到池化，再经过全连接层，最后使用softmax分类输出层

![](C:\Users\Administrator\Desktop\Lenet\图片\lenet.png)

#### 2.2卷积层

卷积层是卷积神经网络的核心。在图像里我们提到的卷积一般是二维卷积，即离散二维滤波器（也称作卷积核）与二维图像做卷积操作，简单的讲是二维滤波器滑动到二维图像上所有位置，并在每个位置上与该像素点及其领域像素点做内积。卷积操作被广泛应用与图像处理领域，不同卷积核可以提取不同的特征

~~~python
tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
~~~



![](C:\Users\Administrator\Desktop\Lenet\图片\Convolution_schematic.gif)

#### 2.3池化层

- pooling，将一个区域中的信息压缩成一个值，完成信息的抽象

~~~python
pool1 = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
~~~



![](C:\Users\Administrator\Desktop\Lenet\图片\池化.png)

最大池化与均值池化

- 区域最大值/均值
- 选取区域最大值（max pooling）,能更好保留纹理特征
- 选取区域均值（mean pooling），能保留整体数据的特征

![](C:\Users\Administrator\Desktop\Lenet\图片\池化_4.png)

#### 2.4常见的激活函数

sigmoid激活函数
$$
f(x)=\operatorname{sigmoid}(x)=\frac{1}{1+e^{-x}}
$$


tanh激活函数
$$
f(x)=\tanh (x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}
$$


ReLU激活函数
$$
f(x)=\max (0, x)
$$

#### 2.5搭建模型的完整代码

~~~Python
import tensorflow as tf
 
INPUT_NODE = 784         
OUTPUT_NODE =10         
                        
IMAGE_SIZE =28           
NUM_CHANNELS = 1          
NUM_LABELS = 10           
'''第一层卷积层的尺寸和深度'''
C0NV1_DEEP = 32
C0NV1_SIZE = 5
'''第二层卷积层的尺寸和深度'''
CONV2_DEEP = 64
CONV2_SIZE = 5
'''全连接层的节点个数'''
FC_SIZE = 512
 
 
def inference(input_tensor, train, regularizer):
    '''
    卷积神经网络的前向传播过程
    @ train 用于区分训练和测试过程
    @ input_tensor 输入变量 四维
    '''
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight",
                                        [C0NV1_SIZE, C0NV1_SIZE, NUM_CHANNELS, C0NV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [C0NV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    '''pool1'''
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
 
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight", 
                                       [CONV2_SIZE,CONV2_SIZE,C0NV1_DEEP,CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))   
        
        with tf.name_scope("layer4-pool2"):
            pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_shape = pool2.get_shape().as_list() # 获取pool2的输出
            '''获取的pool_shape包含batch_size层'''
            nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
            # 通过tf.reshape函数将第四层的输出变成一个batch的向量。
            reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
 
        with tf.variable_scope("layer5-fc1"):
            fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            '''add_to_collection函数将一个张量加入一个集合'losses' '''
            if regularizer != None:
                tf.add_to_collection('losses', regularizer(fc1_weights))
            fc1_biases = tf.get_variable('bias', [FC_SIZE], initializer=tf.constant_initializer(0.1))
            fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
            if train:
                fc1 = tf.nn.dropout(fc1, 0.5)
 
        with tf.variable_scope('layer6-fc2'):
            fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            if regularizer != None:
                tf.add_to_collection('losses', regularizer(fc2_weights))
            fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
            logit = tf.matmul(fc1, fc2_weights) + fc2_biases
        return logit
 
~~~



### 3-实现算法的大致描述（介绍该算法的思路）

> 1. 加载MNIST数据集，并对标签进行one-hot编码
> 2. 将reshape后的数据输入LeNet模型进行特征提取、分类
> 3. 调整参数，使得结果最优

#### 3.1训练代码

~~~Python
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
import  mnist_inference
 

BATCH_SIZE = 100                                     
LEARNING_RATE_BASE = 0.01     
LEARNING_RATE_DECAY =0.99     
REGULARAZTION_RATE = 0.0001      #正则化项的权重
TRAINING_STEPS = 8000
 
def train( mnist ):
    '''定义输入输出placeholder'''
    x = tf.placeholder(tf.float32, [BATCH_SIZE,
                                    mnist_inference.IMAGE_SIZE,
                                    mnist_inference.IMAGE_SIZE,
                                    mnist_inference.NUM_CHANNELS], name = 'x-input1')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE] , name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer( REGULARAZTION_RATE ) #返回一个可以计算l2正则化项的函数
    '''前向传播过程'''
    y = mnist_inference.inference(x,True,regularizer)
    
    '''损失函数'''
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1)))
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    
    train_step = tf.train.AdamOptimezer(learning_rate).minimize(loss)
    '''计算准确率'''
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 初始化所有变量
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print("------------开始训练--------------")
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch( BATCH_SIZE )
            # 类似地将输入的训练数据格式调整为一个四维矩阵,并将这个调整后的数据传入sess.run过程
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.NUM_CHANNELS))
            train_accuracy = sess.run(accuracy,feed_dict={x: reshaped_xs, y_: ys})
            if i % 1000 == 0:
                print ( "After %d training step (s) , loss on training batch is %g." % (i, train_accuracy))
                saver.save(sess,'./model/model.ckp',global_step = global_step)
        print("------------------训练结束-----------------")
# 主程序入口
def main(argv=None):
    '''
    主程序入口
    声明处理MNIST数据集的类，这个类在初始化时会自动下载数据
    '''
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    if mnist != None:
        print("-------------数据加载完毕------------------")
    train(mnist)
    
if __name__	 == '__main__':
    tf.app.run ()
~~~



### 4-验证并简要分析评价算法得出的结果

#### 4.1验证代码

> 加载训练的模型参数，leNet-5模型的准确率能够达到99%

~~~Python
import time
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
 
import mnist_inference
import mnist_train
'''每10秒加载一次模型，并测试最新的正确率'''
EVAL_INTERVAL_SECS = 10
 
def evaluate( mnist ):
    with tf.Graph().as_default() as g:   # 将默认图设为g
        '''定义输入输出的格式'''
        x = tf.placeholder(tf.float32, [mnist.validation.images.shape[0],
                                        mnist_inference.IMAGE_SIZE,
                                        mnist_inference.IMAGE_SIZE,
                                        mnist_inference.NUM_CHANNELS], name='x-input1')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
 
        xs = mnist.validation.images
        # 类似地将输入的测试数据格式调整为一个四维矩阵
        reshaped_xs = np.reshape(xs, (mnist.validation.images.shape[0],
                                      mnist_inference.IMAGE_SIZE,
                                      mnist_inference.IMAGE_SIZE,
                                      mnist_inference.NUM_CHANNELS))
        validate_feed = {x: reshaped_xs, y_: mnist.validation.labels}
        '''前向传播测试，不需要正则项'''
        y = mnist_inference.inference(x,None, None)
 
        #使用tf.argmax(y, 1)就可以得到输入样例的预测类别
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # 首先将一个布尔型的数组转换为实数，然后计算平均值
        #True为1，False为0
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver(variable_to_restore) 

        for i in range(2):    
            with tf.Session() as sess:
                # 会通过checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state( "./model")
                if ckpt and ckpt.model_checkpoint_path:
                    #加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    #得到所有的滑动平均值
                    #通过文件名得到模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict = validate_feed)           #使用此模型检验
                    #没有初始化滑动平均值，只是调用模型的值，inference只是提供了一个变量的接口，完全没有赋值
                    print("After %s training steps, validation accuracy = %g" %(global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
 
def main( argv=None ):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    evaluate(mnist)
 
if __name__=='__main__':
    tf.app.run()
~~~

## 参考

1. 《Tensorflow 实战Google深度学习框架》

2. [paddlePaddle官方文档【图片分类】](http://www.paddlepaddle.org.cn/documentation/docs/zh/1.4/beginners_guide/basics/image_classification/index.html)

3. [吴恩达深度学习（第四门课-卷积神经网络【第一周】）](https://mooc.study.163.com/course/2001281004?tid=2001392030&_trace_c_p_k2_=2b2183fb56f24f25994a65c20704fad1#/info)