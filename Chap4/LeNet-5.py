# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 14:53:00 2018

@author: Administrator
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

input_node = 784
output_node = 10
BATCH_SIZE = 100
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

#layer 1
CONV1_DEEP = 32
CONV1_SIZE = 5
#layer 2
CONV2_DEEP = 64
CONV2_SIZE = 5
#Full-connect
FC_SIZE = 512

Model_save_path = "C:\\Users\\Administrator\\tensor_flow\\Chap4\\"
Model_name = "DNN_model.ckpt"


def mnist_inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_w = tf.get_variable(
                "weight", [CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_b = tf.get_variable(
                "bias",[CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(
                input_tensor, conv1_w,strides=[1,1,1,1],padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    with tf.name_scope('layer2_pool'):
        pool1 = tf.nn.max_pool(
                relu1, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    with tf.variable_scope('layer3-conv2'):
        conv2_w = tf.get_variable(
                "weight", [CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_b = tf.get_variable(
                "bias",[CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(
                pool1, conv2_w,strides=[1,1,1,1],padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    with tf.name_scope('layer4_pool2'):
        pool2 = tf.nn.max_pool(
                relu2, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
    with tf.variable_scope('layer5-fc1'):
        fc1_w = tf.get_variable(
                "weight", [nodes, FC_SIZE],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('loss', regularizer(fc1_w))
        fc1_b = tf.get_variable(
                "bias",[FC_SIZE], initializer=tf.constant_initializer(0.0))


        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
        if train is not None: 
            fc1 = tf.nn.dropout(fc1, 0.5)
            
    with tf.variable_scope('layer6-fc2'):
        fc2_w = tf.get_variable(
                "weight", [FC_SIZE, NUM_LABELS],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('loss', regularizer(fc2_w))
        fc2_b = tf.get_variable(
                "bias",[NUM_LABELS], initializer=tf.constant_initializer(0.0))

        logit = tf.matmul(fc1, fc2_w) + fc2_b
    
    return logit
        
def train_model(data):
    '''train model in dataset'''
    
    learning_rate = 0.008
    rate_decay = 0.99
    regular_rate = 0.001
    training_step = 5000
    moving_avg_decay = 0.99
    batch_size = 100
    x_,y_ = data.train.next_batch(batch_size)
    n_y = y_.shape[1]
#    n_h = 250
    
    xs = tf.placeholder(tf.float32,[
        BATCH_SIZE,
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNELS],
        name='x_inp')
    ys = tf.placeholder(tf.float32,[None, n_y], name='y_inp')

            
    reg = tf.contrib.layers.l2_regularizer(regular_rate)
    y_pred = mnist_inference(xs,'train',reg)
    
    global_step = tf.Variable(0, trainable=False)
    var_avg = tf.train.ExponentialMovingAverage(moving_avg_decay,global_step)
    var_avg_op = var_avg.apply(tf.trainable_variables())
    
    #specify the train step and opt
    cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y_pred, labels=tf.argmax(ys,1))
    cross_entropy = tf.reduce_mean(cross_ent)
    
    
    loss = cross_entropy + tf.add_n(tf.get_collection('loss'))
    learning_rate = tf.train.exponential_decay(
            learning_rate, global_step, data.train.num_examples/batch_size,
            rate_decay)
    
    """define the loss with softmax"""
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            loss, global_step=global_step)
    with tf.control_dependencies([train_step,var_avg_op]):
        train_op = tf.group(train_step, var_avg_op)
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
         
#        validate_feed = {xs:data.validation.images,
#                         ys:data.validation.labels}
#        
#        test_feed = {xs:data.test.images,ys:data.test.labels}
        
        for i in range(training_step):
            X, Y = data.train.next_batch(batch_size)
                
            reshaped_X = np.reshape(X, (BATCH_SIZE,
                IMAGE_SIZE,
                IMAGE_SIZE,
                NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss,global_step],
                                           feed_dict={xs:reshaped_X,ys:Y})
            if i % 1000 == 0:
#                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print("After %d steps, accuracy using avg_model is %g" %
                      (step, loss_value))
#                saver.save(sess,os.path.join(Model_save_path,Model_name),
#                           global_step=global_step)
def main(argv=None):
    
    mnist = input_data.read_data_sets("mnist_data", one_hot=True)
    print("training data size:", mnist.train.num_examples)
    train_model(mnist)

if __name__ == '__main__':
    tf.app.run()