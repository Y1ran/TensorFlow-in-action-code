# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 21:32:47 2018

@author: Administrator
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def forward_prop(input_data, avg_class,n_x,n_h,n_y,reuse=True):
    '''without avg_class, it use the variable'''
    if avg_class == None:
        with tf.variable_scope('layer1', reuse=reuse):
            w = tf.get_variable("weight", [n_x,n_h])
            b = tf.get_variable("bias", [n_h])
            a1 = tf.nn.relu(tf.matmul(input_data,w) + b)
        with tf.variable_scope('layer2', reuse=reuse):
            w = tf.get_variable("weight", [n_h,n_y])
            b = tf.get_variable("bias", [n_y])
            z2 = tf.matmul(a1, w) + b
    else:
        with tf.variable_scope('layer1', reuse=reuse):
            w = tf.get_variable("weight", [n_x,n_h])
            b = tf.get_variable("bias", [n_h])
        #first use avg_class, then forward prop
            a1 = tf.nn.relu(tf.matmul(input_data,avg_class.average(w)) + 
                        avg_class.average(b))
        with tf.variable_scope('layer2', reuse=reuse):
            w = tf.get_variable("weight", [n_h,n_y],)
            b = tf.get_variable("bias", [n_y])
            z2 = tf.matmul(a1,avg_class.average(w)) + avg_class.average(b)
        
    return z2

def init_parameters(n_x, n_h, n_y, reuse=False):
    
    with tf.variable_scope('layer1', reuse=reuse):
        w = tf.get_variable("weight", [n_x,n_h],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable("bias", [n_h],
                            initializer=tf.constant_initializer(0.0))
    with tf.variable_scope('layer2', reuse=reuse):
        w = tf.get_variable("weight", [n_h,n_y],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable("bias", [n_y],
                            initializer=tf.constant_initializer(0.0))    
#    w1 = tf.Variable(tf.truncated_normal([n_x, n_h], stddev=0.1))
#    b1 = tf.Variable(tf.constant(0.1, shape=[n_h]))
#    w2 = tf.Variable(tf.truncated_normal([n_h, n_y], stddev=0.1))
#    b2 = tf.Variable(tf.constant(0.1, shape=[n_y]))
#    
#    paras = [w1,b1,w2,b2]
#    return paras
    

            
def train_model(data):
    '''train model in dataset'''
    
    learning_rate = 0.8
    rate_decay = 0.99
    regular_rate = 0.001
    training_step = 5000
    moving_avg_decay = 0.99
    batch_size = 100
    
    X, Y = data.train.next_batch(batch_size)
    
    print("X shape: {}".format(X.shape))
    
    n_x = X.shape[1]
    n_y = Y.shape[1]
    n_h = 250
    
    xs = tf.placeholder(tf.float32,[None, n_x], name='x_inp')
    ys = tf.placeholder(tf.float32,[None, n_y], name='y_inp')
    
    init_parameters(n_x,n_h,n_y)
    
    y_pred = forward_prop(xs, None, n_x,n_h,n_y)
    
    global_step = tf.Variable(0, trainable=False)
    var_avg = tf.train.ExponentialMovingAverage(moving_avg_decay,global_step)
    var_avg_op = var_avg.apply(tf.trainable_variables())
    y_avg = forward_prop(xs, var_avg,n_x,n_h,n_y)
    
    cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y_pred, labels=tf.argmax(ys,1))
    cross_entropy = tf.reduce_mean(cross_ent)
    
    # use L2 regularization
    reg = tf.contrib.layers.l2_regularizer(regular_rate)
    with tf.variable_scope('layer1', reuse=True):
        w = tf.get_variable("weight", [n_x,n_h])
        reg1 = reg(w)
    with tf.variable_scope('layer2', reuse=True):
        w = tf.get_variable("weight", [n_h,n_y],)
        reg2 = reg(w)
    regular = reg1 + reg2

    """define the loss with softmax"""
    loss = cross_entropy + regular
    learning_rate = tf.train.exponential_decay(
            learning_rate, global_step, data.train.num_examples/batch_size,
            rate_decay)
    #specify the train step and opt
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            loss, global_step=global_step)
    train_op = tf.group(train_step, var_avg_op)
    
    correct = tf.equal(tf.argmax(y_avg,1),tf.argmax(ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
         
        validate_feed = {xs:data.validation.images,
                         ys:data.validation.labels}
        
        test_feed = {xs:data.test.images,ys:data.test.labels}
        
        for i in range(training_step):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print("After %d steps, accuracy using avg_model is %g" %
                      (i, validate_acc))
            
            x_,y_ = data.train.next_batch(batch_size)
            sess.run(train_op,feed_dict={xs:x_,ys:y_})
    
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d steps, test accuracy is: %g" %(training_step, test_acc))
        saver.save(sess,r'C:\Users\Administrator\tensor_flow\Chap3\DNN_v1.ckpt')
#
#def main(argv=None):
#    
#    mnist = input_data.read_data_sets("mnist_data", one_hot=True)
#    print("training data size:", mnist.train.num_examples)
#    
#    train_model(mnist)
    
if __name__ =='__main__':
    mnist = input_data.read_data_sets("mnist_data", one_hot=True)
    print("training data size:", mnist.train.num_examples)
    
    train_model(mnist)
    