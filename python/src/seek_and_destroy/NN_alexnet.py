#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
AlexNet Tensorflow Implmentation for AUVSI SUAS
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
from datetime import datetime
import tensorflow as tf
import numpy as np
import pickle
import random
import cv2
import sys
import os

__author__ = "gndctrl2mjrtm"

"""
===============================================================================
Dataset Preperation
===============================================================================
"""

class Dataset():

    def __init__(self,root):
        """Parse dataset from root directory of subdirectories into array"""
        # Debug root path arg
        if not os.path.exists(root):
            raise SystemError("Root directory does not exist:{}".format(
                root))
        else:
            if not os.path.isdir(root):
                raise SystemError("Root directory arg is not a directory")

        # Whether to display dataset being processed
        self.CONFIG_DISPLAY   = False

        # Whether to shuffle the dataset before being output
        self.CONFIG_SHUFFLE   = True

        # Number of examples of each classifcation
        self.n_datapoints     = 100000

        # Root directory of the folders of 
        self.root             = root

        # Ratio of number of testing datapoints to training datapoints
        self.test_train_ratio = 0.3

        # Names of the labels
        self.label_names      = os.listdir(root)
        for n in self.label_names:
            if not os.path.isdir(os.path.join(self.root,n)):
                self.label_names.remove(n)

        # Save file paths
        self.data_path        = "data.p"
        self.label_path       = "labels.p"
        
        # 1 Hot Envdoing Vector generation
        HEV_fnct              = lambda a,b: 1. if a==b else 0.
        self.fetch_1HEV       = lambda z: [HEV_fnct(n,z) for n in self.label_names]

    #--------------------------------------------------------------------------

    def process_dataset(self):
        """Load the dataset from root into np arrays"""
        self.data   = []
        self.labels = []

        # Find frame data
        for class_dir in self.label_names:
            datapoint_count = 0
            print(class_dir)
            target_dir = os.path.join(self.root,class_dir)

            if os.path.isdir(target_dir):

                for img in os.listdir(os.path.join(self.root,class_dir)):
                    if not (datapoint_count >= self.n_datapoints):
                        #print(datapoint_count)
                        # If image file
                        if (img.split('.')[1] in ['jpg','JPEG','png']):
                            full_path = os.path.join(target_dir,img)
                            frame     = np.asarray(cv2.imread(full_path))
     
                            # Display frame per config settings
                            if self.CONFIG_DISPLAY:
                                cv2.imshow("frame",frame)
                                cv2.waitKey(1)

                            # Append data and release memory
                            self.data.append(frame)
                            # print(self.data.shape)
                            label = self.fetch_1HEV(class_dir)
                            #print(label)
                            self.labels.append(label)
                            del frame
                            datapoint_count += 1
                    else:
                        continue
        # Shuffle dataset
        if self.CONFIG_SHUFFLE:
            self.shuffle()

    #--------------------------------------------------------------------------

    def shuffle(self):
        """Shuffle dataset and labels"""
        index = list(range(len(self.data)))
        random.shuffle(index)
        self.data = [self.data[i] for i in index]
        self.labels = [self.labels[i] for i in index]

    #--------------------------------------------------------------------------

    def save_data(self):
        """Save data and labels to respective files"""
        with open(self.data_path,"wb") as data_file:
            pickle.dump(self.data,data_file)
        with open(self.label_path,"wb") as labels_file:
            pickle.dump(self.labels,labels_file)

    #--------------------------------------------------------------------------

    def load_data(self):
        """Load preprocessed data from pickle file"""
        with open(self.data_path,"rb") as data_file:
            self.data = pickle.load(data_file)
        
        for i,n in enumerate(self.data):
            self.data[i] = cv2.resize(n,(227,227))
        
        with open(self.label_path,"rb") as labels_file:
            self.labels = pickle.load(labels_file)

    #--------------------------------------------------------------------------

    def training_data(self):
        """Return training data"""
        return self.data[:int(len(self.data)*(1.-self.test_train_ratio))]

    #--------------------------------------------------------------------------

    def training_labels(self):
        """Return training labels"""
        return np.asarray(self.labels[:int(len(self.data)*(
            1.-self.test_train_ratio))])

     #--------------------------------------------------------------------------

    def testing_data(self):
        """Return testing data"""
        return np.asarray(self.data[int(len(self.data)*(
            1.-self.test_train_ratio)):])

    #--------------------------------------------------------------------------

    def testing_labels(self):
        """Return testing labels"""
        return np.asarray(self.labels[int(len(self.data)*(
            1.-self.test_train_ratio)):])





class TFAlexNet(object):

    def __init__(self,training_data,training_labels,testing_data,testing_labels):
        self.verify_input_dataset(training_data,"training_data")
        self.training_data    = training_data

        self.training_labels  = training_labels

        self.verify_input_dataset(testing_data,"testing_data")
        self.testing_data     = testing_data

        self.testing_labels   = testing_labels

        self.use_tensorboard  = True

        # Training Parameters
        self.learning_rate    = 0.001
        self.num_steps        = 100000
        self.batch_size       = 128
        self.display_step     = 100
        self.epochs           = 1
        self.leaky_relu_alpha = 0.2
        self.dropout_prob     = 0.5

        self.model_path       = "/tmp/model.ckpt"
        
        now                   = datetime.now()

        self.logs_path        = "/tmp/tensorflow_logs/{}/".format(now.strftime("%Y%m%d-%H%M%S"))
        self.num_classes      = len(training_labels[0])
        self.x                = tf.placeholder(tf.float32, [None,227,227,3],name='x')
        self.y                = tf.placeholder(tf.float32, [None, self.num_classes],name='y')
        self.keep_prob        = tf.placeholder(tf.float32,name='keep_prob') # dropout_layer (keep probability)

        with tf.name_scope('Weights'):
            self.weights      = {
                'wc1': tf.Variable(
                    tf.random_normal([5,5,3,32]),name='wc1'),
                'wc2': tf.Variable(
                    tf.random_normal([3,3,32,64]),name='wc2'),
                'wc3': tf.Variable(
                    tf.random_normal([2,2,64,384]),name='wc3'),
                'wc4': tf.Variable(
                    tf.random_normal([2,2,384,384]),name='wc4'),
                'wc5': tf.Variable(
                    tf.random_normal([2,2,384,256]),name='wc5'),
                'wf1': tf.Variable(
                    tf.random_normal([2304,4096]),name='wf1'),
                'wf2': tf.Variable(
                    tf.random_normal([4096,4096]),name='wf2'),
                'wf3': tf.Variable(
                    tf.random_normal([4096,self.num_classes]),name='wf3'),
            }
        with tf.name_scope('Biases'):
            self.biases       = {
                'bc1': tf.Variable(
                    tf.random_normal([32]),name='bc1'),
                'bc2': tf.Variable(
                    tf.random_normal([64]),name='bc2'),
                'bc3': tf.Variable(
                    tf.random_normal([384]),name='bc3'),
                'bc4': tf.Variable(
                    tf.random_normal([384]),name='bc4'),
                'bc5': tf.Variable(
                    tf.random_normal([256]),name='bc5'),
                'bf1': tf.Variable(
                    tf.random_normal([4096]),name='bf1'),
                'bf2': tf.Variable(
                    tf.random_normal([4096]),name='bf2'),
                'bf3': tf.Variable(
                    tf.random_normal([self.num_classes]),name='bf3')
            }

    #--------------------------------------------------------------------------

    def verify_input_dataset(self,dataset,name):
        """Verify that the dataset has the correct data types"""
        if not isinstance(dataset,(np.ndarray,list)):
            raise TypeError("Incorrect {} arg type:{}".format(name,
                type(dataset).__name__))
        for i,n in enumerate(dataset):
            if not isinstance(n,np.ndarray):
                raise TypeError("{}, Incorrect type:{} index:{}".format(
                    name,type(n).__name__),i)
            if not (n.ndim == 3):
                raise Exception("Incorrect n-dim:{}, index:{}:{}".format(
                    name,i,n.ndim))
            if not (np.issubdtype(n.dtype,np.number)):
                raise Exception("{}, Non-number ndarray:{}".format(
                    name,n.dtype.__name__))

    #--------------------------------------------------------------------------

    def alexnet(self,x):
        conv1 = self.conv2d(x,self.weights["wc1"],self.biases["bc1"],
            x_stride=4,y_stride=4,padding="VALID",name="conv1")
        lrn1 = self.local_response_normalization(conv1,2,2e-05,0.75,"norm1")
        pool1 = self.maxpool_layer_2d(lrn1,k_width=3,k_height=3,x_stride=2,
            y_stride=2,padding="VALID",name="pool1")

        conv2 = self.conv2d(pool1,self.weights["wc2"],self.biases["bc2"],
            x_stride=1,y_stride=1,padding="VALID",name="conv2")
        lrn2 = self.local_response_normalization(conv2,2,2e-05,0.75,"norm2")
        pool2 = self.maxpool_layer_2d(lrn2,k_width=3,k_height=3,x_stride=2,
            y_stride=2,padding="VALID",name="pool2")

        conv3 = self.conv2d(pool2,self.weights["wc3"],self.biases["bc3"],
            x_stride=1,y_stride=1,padding="VALID",name="conv3")

        conv4 = self.conv2d(conv3,self.weights["wc4"],self.biases["bc4"],
            x_stride=1,y_stride=1,padding="VALID",name="conv4")

        conv5 = self.conv2d(conv3,self.weights["wc5"],self.biases["bc5"],
            x_stride=1,y_stride=1,padding="VALID",name="conv5")
        pool5 = self.maxpool_layer_2d(conv5,k_width=3,k_height=3,x_stride=2,
            y_stride=2,padding="VALID",name="pool5")

        pool_shape = pool5.get_shape().as_list()
        fully_connected_input = tf.reshape(pool5,[-1,
            pool_shape[1]*pool_shape[2]*pool_shape[3]])

        fully_connected1 = self.fully_connected_layer(fully_connected_input,
            self.weights["wf1"],self.biases["bf1"],name="ff1")
        fully_connected1 = tf.nn.leaky_relu(fully_connected1,
            alpha=self.leaky_relu_alpha)
        dropout1 = self.dropout_layer(fully_connected1)

        fully_connected2 = self.fully_connected_layer(dropout1,
            self.weights["wf2"],self.biases["bf2"],name="ff2")
        fully_connected2 = tf.nn.leaky_relu(fully_connected2,
            alpha=self.leaky_relu_alpha)
        dropout2 = self.dropout_layer(fully_connected2)

        fully_connected3 = self.fully_connected_layer(dropout2,
            self.weights["wf3"],self.biases["bf3"],name="ff3")
        fully_connected3 = tf.nn.leaky_relu(fully_connected3,
            alpha=self.leaky_relu_alpha)

        return fully_connected3

    #--------------------------------------------------------------------------

    # Create some wrappers for simplicity
    def conv2d(self,x,weights,bias,x_stride=1,y_stride=1,padding="SAME",name='hidden'):
        # Conv2D wrapper, with bias and relu activation
        with tf.name_scope(name):
            x = tf.nn.conv2d(x,weights,strides=[1,y_stride,x_stride,1],padding=padding)
            x = tf.nn.bias_add(x,bias)
        return tf.nn.leaky_relu(x,alpha=self.leaky_relu_alpha)

    #--------------------------------------------------------------------------

    def fully_connected_layer(self,x,weights,bias,name='hidden'):
        with tf.name_scope(name):
            return tf.add(tf.matmul(x,weights),bias)

    #--------------------------------------------------------------------------

    def dropout_layer(self,x,name=None):
        return tf.nn.dropout(x,self.keep_prob,name)

    #--------------------------------------------------------------------------

    def local_response_normalization(self,x,radius,alpha,beta,name=None,bias=1.0):
        return tf.nn.local_response_normalization(x,depth_radius=radius,
            alpha=alpha,beta=beta,bias=bias,name=name)

    #--------------------------------------------------------------------------

    def maxpool_layer_2d(self,x, k_width=2,k_height=2,x_stride=2,y_stride=2,
            padding="SAME",name="hidden"):
        # MaxPool2D wrapper
        with tf.name_scope(name):
            return tf.nn.max_pool(x, ksize=[1,k_height,k_width,1],
                strides=[1,y_stride,x_stride,1],padding=padding)

    #--------------------------------------------------------------------------

    def init_cost(self):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_constant = 0.01  # Choose an appropriate one.
        print(self.logits,self.y)
        sftmx = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits,labels=self.y))
        return sftmx+(reg_constant*sum(reg_losses))

    #--------------------------------------------------------------------------

    def init_network(self):
        """Initialize network variables and architecture"""
        # Declare model
        with tf.name_scope('Model'):
            self.logits = self.alexnet(self.x)
            print(self.logits)
            self.prediction = tf.nn.softmax(self.logits)


        with tf.name_scope('Cost'):
            self.cost = self.init_cost()

        with tf.name_scope('SGD'):
            self.optimizer_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.optimizer = self.optimizer_op.minimize(self.cost)

        with tf.name_scope('Accuracy'):
            correct_pred = tf.equal(tf.argmax(self.prediction,1),
                tf.argmax(self.y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

        self.init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()

        tf.summary.scalar("cost",self.cost)
        tf.summary.scalar("accuracy",self.accuracy)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.name,var)

        self.merged_summary_op = tf.summary.merge_all()

    #--------------------------------------------------------------------------

    def fetch_batch(self,data,labels,batch_size):
        """Return randomized batches of data"""
        index = list(range(len(data)))
        random.shuffle(index)
        index = index[:batch_size]
        batch_x = [data[i] for i in index]
        batch_y = [labels[i] for i in index]
        return batch_x,batch_y

    #--------------------------------------------------------------------------

    def train(self,sess):
        """Train model with batch learning"""
        if self.use_tensorboard:
            summary_writer = tf.summary.FileWriter(self.logs_path,
                                graph=tf.get_default_graph())
            # Retrain the model
            for epoch in range(self.epochs):
                for step in range(self.num_steps):

                    # Fetch data batches
                    batch_x,batch_y = self.fetch_batch(self.training_data,
                        self.training_labels,self.batch_size)

                    # Run optimization
                    if self.use_tensorboard:
                        _,summary = sess.run([self.optimizer,
                            self.merged_summary_op],
                            feed_dict={self.x: batch_x, self.y: batch_y,
                            self.keep_prob: self.dropout_prob})
                    else:
                        _,= sess.run([self.optimizer],
                            feed_dict={self.x: batch_x, self.y: batch_y,
                            self.keep_prob: self.dropout_prob})

                    # Update Tensorboard data
                    if self.use_tensorboard:
                        summary_writer.add_summary(summary,
                            (epoch*self.num_steps + step))

                    # Print training data step updates
                    if (step % self.display_step == 0) or (step == 1):
                        loss,acc = sess.run([self.cost, self.accuracy],
                            feed_dict={self.x:batch_x,self.y:batch_y,
                            self.keep_prob:self.dropout_prob})

                        print("Step:{}\tLoss:{}\tAccuracy:{}".format(
                            step,loss,acc),file=sys.stderr)

            # Fetch testing data batches
            test_x,test_y = self.fetch_batch(self.testing_data,
                self.testing_labels,25)

            # Print testing data accuracy
            print("Testing Accuracy:{}".format(sess.run(self.accuracy,
                feed_dict={self.x: test_x,self.y:test_y,
                self.keep_prob: 1.0})),file=sys.stderr)

            # Save the retrained model
            self.save_path = self.saver.save(sess,self.model_path)
            print("Model saved in file:{}".format(self.save_path),
                file=sys.stderr)

            # Print tensorboard command
            if self.use_tensorboard:
                print("Run the command line:\n--> tensorboard --logdir={} "\
                  "\nOpen http://0.0.0.0:6006/".format(self.logs_path))

    #--------------------------------------------------------------------------

    def learn(self):
        with tf.Session() as sess:
            # Initialize variables
            sess.run(self.init)
            self.train(sess)

    #--------------------------------------------------------------------------

    def continue_training(self):
        # Check trained model path arg
        if not isinstance(self.model_path,str):
            raise TypeError("Incorrect file path type:{}".format(type(
                self.save_path).__name__))
        
        if not os.path.exists(self.model_path):
            raise Exception("Model path does not exist:{}".format(
                self.save_path))

        with tf.Session() as sess:
            # Initialize variables
            sess.run(self.init)

            # Restore saved model
            self.saver.restore(sess,self.model_path)

            print("Model restored from file:{}".format(self.model_path),
                file=sys.stderr)

            # Continue training model
            self.train(sess)

    #--------------------------------------------------------------------------

    def evaluate_video(self):
        with tf.Session() as sess:
            # Initialize variables
            sess.run(self.init)

            a,b = 100,200

            # Restore model weights from previously saved model
            self.saver.restore(sess,self.model_path)

            video_data = cv2.VideoCapture(0)

            while True:
                _,frame = video_data.read()

                subframe = frame[a:b,a:b]
                frame = cv2.rectangle(frame,(a,a),(b,b),[0,255,0])
                cv2.imshow("frame",frame)

                subframe = cv2.cvtColor(subframe,cv2.COLOR_BGR2GRAY)
                _,subframe = cv2.threshold(subframe,127,255,cv2.THRESH_BINARY_INV)
                cv2.imshow("subframe",subframe)
                subframe = cv2.resize(subframe,(28,28))
                subframe = np.reshape(subframe,(-1,784))
                subframe = subframe/256.

                pred     = sess.run(self.prediction,
                    feed_dict={self.x:subframe,self.keep_prob:1.0})
                scores = sess.run(self.logits,
                    feed_dict={self.x:subframe,self.keep_prob:1.0})
                print(pred[0])
                if 1.0 in pred:
                    classification = np.where(pred==1.0)[1][0]
                    print(classification)
                cv2.waitKey(1)

    #--------------------------------------------------------------------------

    def preprocess_frame(self,frame):
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # frame = np.uint8(frame/np.average(frame))
        frame = cv2.normalize(frame,frame,0,255, cv2.NORM_MINMAX)
        #print(frame)
        return frame

    #--------------------------------------------------------------------------

    def evaluate_frame(self,sess,frame):
        subframe   = self.preprocess_frame(frame)

        cv2.imshow("subframe",subframe)
        cv2.waitKey(1)

        subframe   = cv2.resize(subframe,(28,28))
        subframe   = np.reshape(subframe,(-1,784))
        subframe   = subframe/256.
        pred       = sess.run(self.prediction,
            feed_dict={self.x:subframe,self.keep_prob:1.0})
        scores     = sess.run(self.logits,
            feed_dict={self.x:subframe,self.keep_prob:1.0})
        print(pred[0])

#------------------------------------------------------------------------------

def test():
    cnn = TFConvNN()
    cnn.init_network()

    with tf.Session() as sess:
        # Initialize variables
        sess.run(cnn.init)

        # Restore model weights from previously saved model
        cnn.saver.restore(sess,cnn.model_path)
        video_data = cv2.VideoCapture(0)
        while True:
            _,frame = video_data.read()
            cnn.evaluate_frame(sess,frame)
            cv2.waitKey(1)

#------------------------------------------------------------------------------

def prepare_dataset(data):
    output = []
    for n in data:
        n = np.reshape(n,(28,28))
        n = cv2.resize(n,(227,227))
        n = cv2.cvtColor(n,cv2.COLOR_GRAY2BGR)
        #cv2.imshow("data",n)
        #cv2.waitKey(1)
        output.append(np.uint8(255*n))
        del n
    del data
    return output

#------------------------------------------------------------------------------

def main():
    data_root = "/Users/rosewang/Desktop/Programming/generated_data/letters"
    letter_data = Dataset(data_root)

    letter_data.load_data()


    training_data   = letter_data.training_data()
    training_labels = letter_data.training_labels()
    testing_data    = letter_data.testing_data()
    testing_labels  = letter_data.testing_labels()

    cnn = TFAlexNet(training_data,training_labels,testing_data,testing_labels)
    cnn.init_network()
    cnn.learn()

#------------------------------------------------------------------------------

if __name__ == "__main__":
    # dataset_test()
    main()

