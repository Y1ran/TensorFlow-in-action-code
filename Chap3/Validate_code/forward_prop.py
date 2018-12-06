# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 19:08:29 2018

@author: Administrator
"""

import tensorflow as tf

def get_parameters(shape, regularizer):
    
    weight = tf.get_variable("weight",shape,
                        initializer=tf.truncated_normal_initializer(stddev=0.1))

    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weight))
        
    return weight

def forward_prop(input_data,n_x,n_h,n_y, regularizer,reuse=False):
    '''without avg_class, it use the variable'''

    with tf.variable_scope('layer1', reuse=reuse):
        w = get_parameters([n_x,n_h],regularizer)
        b = tf.get_variable("bias", [n_h],initializer=tf.constant_initializer(0.0))
        a1 = tf.nn.relu(tf.matmul(input_data,w) + b)
            
    with tf.variable_scope('layer2', reuse=reuse):
        w = get_parameters([n_h,n_y],regularizer)
        b = tf.get_variable("bias", [n_y],initializer=tf.constant_initializer(0.0))
        z2 = tf.matmul(a1, w) + b
   
        
    return z2