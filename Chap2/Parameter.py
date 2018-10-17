# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 20:50:13 2018

@author: Administrator
"""

import tensorflow as tf

# 初始化参数，标准差为2
init_paras = tf.Variable(tf.random_normal([2,3], stddev=2))
zero_paras = tf.zeros([2,3],tf.int32)
ones_paras = tf.fill([2,3],1.0)

#必须使用会话来进行初始化
sess = tf.Session()
print(sess.run(init_paras.initial_value),
        sess.run(zero_paras),
        sess.run(ones_paras))

#定义两个变量
w1 = tf.Variable(tf.random_normal([2,3], stddev=2, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=2, seed=1))

#X作为常量代表输入数据（L0）
x = tf.constant([[0.7, 0.9]])
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess.run(w1.initializer)
sess.run(w2.initializer)
print(sess.run(tf.matmul(sess.run(a), w2)))
print(sess.run(y))

sess.close()
