#!/bin/env/python
# -*- encoding: utf-8 -*-
# Tensorboard Example for Intel AI Devcloud
from __future__ import print_function
import tensorflow as tf

# MNIST Dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#------------------------------------------------------------------------------
# Variables and Placeholders

learning_rate = 0.001
batch_size = 256
epochs = 10

# Input and Output Layers
x_input = tf.placeholder(tf.float32,[None,784],name='input_layer')
y = tf.placeholder(tf.float32,[None,10],name='output')

# Weights
w1 = tf.Variable(tf.random_normal([784,10]),name='layer1_weights')
w2 = tf.Variable(tf.random_normal([10,10]),name='layer2_weights')

# Biases
b1 = tf.Variable(tf.random_normal([10]),name='layer1_bias')
b2 = tf.Variable(tf.random_normal([10]),name='layer2_bias')

#------------------------------------------------------------------------------
# Tensorflow Operations

# Create perceptron graph and name it 'Multilayer Perceptron'
with tf.name_scope('Multilayer_Perceptron'):
	layer1 = tf.matmul(x_input,w1)+b1
	layer2 = tf.nn.softmax(tf.matmul(layer1,w2)+b2)

with tf.name_scope('L2_Cost'):
    cost = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(layer2,y),2)))
tf.summary.scalar('Cost',cost)


with tf.name_scope('Optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Accuracy of Model
with tf.name_scope('Accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(layer2,1),
    	tf.argmax(y,1)),tf.float32))
tf.summary.scalar('Accuracy',accuracy)

# Combine summaries in an operation
summarizer = tf.summary.merge_all()

# Intialize variables
init = tf.global_variables_initializer()

#path = '/home/u14092/tensorboard_logs/multilayer_perceptron/'
path = './tensorboard_logs/multilayer_perceptron/'
summary = tf.summary.FileWriter(path,graph=tf.get_default_graph())

#------------------------------------------------------------------------------
# Training

with tf.Session() as sess:
	sess.run(init)
	
	n_batches = int(mnist.train.num_examples/batch_size)
	for i in range(epochs):
		for j in range(n_batches):
			x_batch,y_batch = mnist.train.next_batch(batch_size)
			s,_,__ = sess.run([summarizer,cost,optimizer],feed_dict={x_input:x_batch,y:y_batch})
			summary.add_summary(s,i*n_batches+j)
