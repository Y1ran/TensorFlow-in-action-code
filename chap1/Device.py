# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 21:40:30 2018

@author: Administrator
"""

import tensorflow as tf

a = tf.constant([1,2], name = "a")
b = tf.constant([2,2], name = "b")

g = tf.Graph()

with g.device('/gpu:0'):
    result = a + b
    f.GraphKeys.VARIABLES
    tf.GraphKeys.GLOBAL_VARIABLES
    tf.GraphKeys.SUMMARIES
    
    result.shape