# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 15:28:17 2018

@author: Administrator
"""

import tensorflow as tf
import numpy as np

with tf.variable_scope("foo"):
    v = tf.get_variable("w1",[10],initializer=tf.constant_initializer(1.0))

with tf.variable_scope("foo",reuse=True):
    w1 = tf.get_variable("w1",[10])

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
#with tf.Session() as sess:
#    print(sess.run(w1))
print(w1.eval())