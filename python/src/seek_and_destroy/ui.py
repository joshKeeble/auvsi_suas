#!/bin/env/python3
#-*- encoding: utf-8 -*-
"""
===============================================================================
Tensorboard Data Visualization for ESRA IREC
===============================================================================

-------------------------------------------------------------------------------
"""
from __future__ import print_function
from __future__ import division
import tensorflow as tf
import numpy as np
from datetime import datetime
import pickle
import cv2
import sys
import os

'''
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



'''
"""

def dataset_test():
    data_root = "/Users/rosewang/Desktop/Programming/generated_data/letters"
    letter_data = Dataset(data_root)
    
    letter_data.process_dataset()
    letter_data.save_data()
    
    

    letter_data.load_data()


    training_data   = letter_data.training_data()
    training_labels = letter_data.training_labels()
    testing_data    = letter_data.testing_data()
    testing_labels  = letter_data.testing_labels()
    LOG_DIR = 'logs'

    images = tf.Variable(training_data,name='images')

    with tf.Session() as sess:
        saver = tf.train.Saver([images])

        sess.run(images.initializer)
        saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))
    

    for data,label in zip(training_data,training_labels):
        cv2.imshow("data",data)
        print(label)
        cv2.waitKey(0)
"""
    

class TensorboardUI(object):

    def __init__(self):

        now = datetime.now()
        self.logs_path = "/tmp/tensorflow_logs/{}/".format(now.strftime("%Y%m%d-%H%M%S"))
        
        self.test = tf.Variable(tf.random_normal([11,11,3,96]),name='wc1')
        self.init_sensor_data()

    #--------------------------------------------------------------------------

    def init_sensor_datastream(self):
        pass

    #--------------------------------------------------------------------------

    def init_sensor_data(self):
        with tf.name_scope("Sensors"):
            self.temp = tf.placeholder(tf.float32,[1,None],name="temperature")
        tf.summary.scalar("Temperature",self.temp)
        self.init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()

        for var in tf.trainable_variables():
            tf.summary.histogram(var.name,var)

        self.merged_summary_op = tf.summary.merge_all()

    def update_sensor_data(self):
        return tf.add(self.temp,self.temp)

    #--------------------------------------------------------------------------

    def main(self):
        with tf.Session() as sess:
            sess.run(self.init)
            summary_writer = tf.summary.FileWriter(self.logs_path,
                                graph=tf.get_default_graph())
            for i in range(100):
                sess.run(self.update_sensor_data(),feed_dict={self.temp:np.random.uniform(-10,10,(1,1))})
                summary = sess.run(self.merged_summary_op,feed_dict={self.temp:np.random.uniform(-10,10,(1,1))})
                summary_writer.add_summary(summary,(epoch*self.num_steps + step))


if __name__ == "__main__":
    tb = TensorboardUI()
    tb.main()


