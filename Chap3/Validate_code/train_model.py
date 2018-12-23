# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 19:17:52 2018

@author: Administrator
"""

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from forward_prop import *

'''train model in dataset'''
    
Model_save_path = "C:\\Users\\Administrator\\tensor_flow\\Chap3\\"
Model_name = "DNN_model.ckpt"

def train_model(data):
    '''train model in dataset'''
    
    learning_rate = 0.8
    rate_decay = 0.99
    regular_rate = 0.001
    training_step = 5000
    moving_avg_decay = 0.99
    batch_size = 100
    x_,y_ = data.train.next_batch(batch_size)
    n_x = x_.shape[1]
    n_y = y_.shape[1]
    n_h = 250
    
    xs = tf.placeholder(tf.float32,[None, n_x], name='x_inp')
    ys = tf.placeholder(tf.float32,[None, n_y], name='y_inp')
    
    reg = tf.contrib.layers.l2_regularizer(regular_rate)
    y_pred = forward_prop(xs,n_x,n_h,n_y,reg)
    
    global_step = tf.Variable(0, trainable=False)
    var_avg = tf.train.ExponentialMovingAverage(moving_avg_decay,global_step)
    var_avg_op = var_avg.apply(tf.trainable_variables())
    
    #specify the train step and opt
    cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y_pred, labels=tf.argmax(ys,1))
    cross_entropy = tf.reduce_mean(cross_ent)
    
    
    loss = cross_entropy + tf.add_n(tf.get_collection('losses'))
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
            _, loss_value, step = sess.run([train_op, loss,global_step],
                                           feed_dict={xs:X,ys:Y})
            if i % 1000 == 0:
#                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print("After %d steps, accuracy using avg_model is %g" %
                      (step, loss_value))
                saver.save(sess,os.path.join(Model_save_path,Model_name),
                           global_step=global_step)


def main(argv=None):
    
    mnist = input_data.read_data_sets("mnist_data", one_hot=True)
    print("training data size:", mnist.train.num_examples)
    train_model(mnist)

if __name__ == '__main__':
    tf.app.run()