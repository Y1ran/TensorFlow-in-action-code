# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 13:55:54 2018

@author: Administrator
"""

import tensorflow as tf

filter_weight = tf.get_variable(
        'weights',[5, 5, 3, 16],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
bias = tf.get_variable(
        'bias', [16], initializer=tf.constant_initializer(0.1))

conv = tf.nn.conv2d(
        input, filter_weight, strides=[1,1,1,1], padding='SAME')
bias = tf.nn.bias_add(conv, bias)
active_conv = tf.nn.relu(bias)

pool = tf.nn.max_pool(active_conv, ksize=[1,3,3,1],
                      strides=[1,2,2,1], padding='SAME')