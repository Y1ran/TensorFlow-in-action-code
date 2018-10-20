# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 20:16:59 2018

@author: Administrator
"""

import tensorflow as tf
from forward_prop import *
from model_eval import *
from train_model import *
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':
    mnist = input_data.read_data_sets("mnist_data", one_hot=True)
    print("training data size:", mnist.train.num_examples)
    evaluate(mnist)