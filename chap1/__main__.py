# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 21:36:56 2018

@author: Administrator
"""

import tensorflow as tf

#if __name__ == "__main__":
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))