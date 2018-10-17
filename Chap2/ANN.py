# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:20:51 2018

@author: Administrator
"""


import tensorflow as tf
from numpy.random import RandomState

learning_rate = 0.001
batch_size = 8

def loss_func(y, y_):
    
    y = tf.sigmoid(y)
    cross_ent = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10,
                                1.0)) + (1-y_) * tf.log(tf.clip_by_value(1-y,
                                                    1e-10, 1.0)))
    return cross_ent

w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None,2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None,1), name="y-input")

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
cross_ent = loss_func(y, y_)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_ent)

rand = RandomState(1)
data_size = 128
X = rand.rand(data_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1,x2) in X]

with tf.Session() as sess:
    
    
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    print(sess.run(w1), sess.run(w2))
    
#    print(sess.run(y, feed_dict={x:[[0.7, 0.9],[0.1,0.4],[0.5,0.8]]}))

    step = 5000
    for i in range(step):
        start = (i * batch_size) % data_size
        end = min(start + batch_size, data_size)
        
        sess.run(train_step,
                 feed_dict={x:X[start:end], y_:Y[start:end]})
        
        if i % 1000 == 0:
            total_cross_ent = sess.run(
                    cross_ent, feed_dict={x:X, y_:Y})
            print("After %d training step, cross ent is %g" %
                          (i, total_cross_ent))
    
    print(sess.run(w1), sess.run(w2))
        
#def loss_func(y, y_):
#    
#    y = tf.sigmoid(y)
#    cross_ent = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10,
#                                1.0)) + (1-y_) * tf.log(tf.clip_by_value(1-y,
#                                                    1e-10, 1.0)))
#    return cross_ent


    
    