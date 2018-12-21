# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 20:00:58 2018

@author: Administrator
"""

import time 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from forward_prop import *
from train_model import *


def evaluate(data):
    
    batch_size = 100
    x_,y_ = data.train.next_batch(batch_size)
    n_x = x_.shape[1]
    n_y = y_.shape[1]
    n_h = 250
    moving_avg_decay = 0.99
    eval_seconds = 10
    
    xs = tf.placeholder(tf.float32,[None, n_x], name='x_inp')
    ys = tf.placeholder(tf.float32,[None, n_y], name='y_inp')
    
    validate_feed = {xs:data.validation.images,
                         ys:data.validation.labels}
    
    y_pred = forward_prop(xs,n_x,n_h,n_y,None)
    
    correct = tf.equal(tf.argmax(y_pred,1),tf.argmax(ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    
    Model_save_path = "C:\\Users\\Administrator\\tensor_flow\\Chap3\\"
    Model_name = "DNN_model.ckpt"
    variable_average = tf.train.ExponentialMovingAverage(
           moving_avg_decay)
    variable_store = variable_average.variables_to_restore()
    saver = tf.train.Saver(variable_store)
    
    while True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(Model_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                
                global_step = ckpt.model_checkpoint_path\
                        .split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print("After %s steps. validation accuracy is:  %g" %
                      ( global_step,accuracy_score))
            else:
                print("No checkpoint file found")
                return
        time.sleep(eval_seconds)
def main(argv=None):
    
    mnist = input_data.read_data_sets("mnist_data", one_hot=True)
    print("training data size:", mnist.train.num_examples)
    evaluate(mnist)
    