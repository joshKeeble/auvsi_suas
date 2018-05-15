'''
Save and Restore a model using TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
import numpy as np
import cv2

class TFConvNN(object):

    def __init__(self):
        """
        # Parameters
        self.learning_rate = 0.001
        self.batch_size = 100
        self.display_step = 1
        self.model_path = "/tmp/model.ckpt"

        # Network Parameters
        self.n_hidden_1 = 256 # 1st layer number of features
        self.n_hidden_2 = 256 # 2nd layer number of features
        self.n_input = 784 # MNIST data input (img shape: 28*28)
        self.n_classes = 10 # MNIST total classes (0-9 digits)

        # tf Graph input
        self.x = tf.placeholder("float", [None, self.n_input])
        self.y = tf.placeholder("float", [None, self.n_classes])

        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input,self.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1,self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2,self.n_classes]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }
        """
        # Training Parameters
        self.learning_rate = 0.001
        self.num_steps = 300
        self.batch_size = 128
        self.display_step = 10

        self.model_path = "/tmp/model.ckpt"

        # Network Parameters
        self.num_input = 784 # MNIST data input (img shape: 28*28)
        self.num_classes = 10 # MNIST total classes (0-9 digits)
        self.dropout = 0.75 # Dropout, probability to keep units

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, self.num_input])
        self.y = tf.placeholder(tf.float32, [None, self.num_classes])
        self.keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


        # Store layers weight & bias
        self.weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, self.num_classes]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([self.num_classes]))
        }

    #--------------------------------------------------------------------------

    # Create some wrappers for simplicity
    def conv2d(self,x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    #--------------------------------------------------------------------------

    def maxpool2d(self,x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

    #--------------------------------------------------------------------------

    # Create model
    def conv_net(self,x, weights, biases, dropout):
        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer
        conv1 = self.conv2d(x, self.weights['wc1'], self.biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = self.maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = self.conv2d(conv1, self.weights['wc2'],self.biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = self.maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, self.dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])
        return out

    #--------------------------------------------------------------------------

    # Create model
    def multilayer_perceptron(self,x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, self.weights['out']) + self.biases['out']
        return out_layer

    #--------------------------------------------------------------------------

    def init_cost(self):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_constant = 0.01  # Choose an appropriate one.
        sftmx = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits,labels=self.y))
        return sftmx+(reg_constant*sum(reg_losses))

    #--------------------------------------------------------------------------

    def init_network(self):
        with tf.name_scope('Model'):
            self.logits = self.conv_net(self.x,self.weights,self.biases,
                self.keep_prob)
            self.prediction = tf.nn.softmax(self.logits)

        with tf.name_scope('Cost'):
            self.cost = self.init_cost()

        with tf.name_scope('SGD'):
            self.optimizer_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.optimizer = self.optimizer_op.minimize(self.cost)

        with tf.name_scope('Accuracy'):
            correct_pred = tf.equal(tf.argmax(self.prediction, 1),
                tf.argmax(self.y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

        self.init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()

        tf.summary.scalar("cost",self.cost)
        tf.summary.scalar("accuracy",self.accuracy)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.name,var)

        merged_summary_op = tf.summary.merge_all()

    #--------------------------------------------------------------------------

    def learn(self):

        print("Starting 1st session...")
        with tf.Session() as sess:

            sess.run(self.init)

            for step in range(1, self.num_steps+1):
                batch_x, batch_y = mnist.train.next_batch(self.batch_size)
                # Run optimization op (backprop)
                sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 0.8})
                if (step % self.display_step == 0) or (step == 1):
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([self.cost, self.accuracy], feed_dict={self.x: batch_x,
                                                                         self.y: batch_y,
                                                                         self.keep_prob: 1.0})
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))

            print("Optimization Finished!")

            # Calculate accuracy for 256 MNIST test images
            print("Testing Accuracy:", \
                sess.run(self.accuracy, feed_dict={
                    self.x: mnist.test.images[:256],
                    self.y: mnist.test.labels[:256],
                    self.keep_prob: 1.0}))

            # Save model weights to disk
            self.save_path = self.saver.save(sess, self.model_path)
            print("Model saved in file: %s" % self.save_path)
            print("Run the command line:\n" \
                  "--> tensorboard --logdir=/tmp/tensorflow_logs " \
                  "\nThen open http://0.0.0.0:6006/ into your web browser")

    #--------------------------------------------------------------------------

    def restore(self):
        # Running a new session
        print("Starting 2nd session...")
        with tf.Session() as sess:
            # Initialize variables
            sess.run(self.init)

            # Restore model weights from previously saved model
            self.saver.restore(sess,self.model_path)
            print("Model restored from file: %s" % self.save_path)

            # Resume training
            for epoch in range(7):
                avg_cost = 0.
                total_batch = int(mnist.train.num_examples / self.batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x, batch_y = mnist.train.next_batch(self.batch_size)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([self.optimizer,self.cost], feed_dict={
                        self.x: batch_x,self.y: batch_y})
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if not (epoch % self.display_step):
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                        "{:.9f}".format(avg_cost))
            print("Second Optimization Finished!")

            # Test model
            correct_prediction = tf.equal(tf.argmax(self.pred, 1),
                tf.argmax(self.y,1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
            print("Accuracy:", accuracy.eval(
                {self.x: mnist.test.images, self.y: mnist.test.labels}))

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
        '''
        if 1.0 in pred:
            classification = np.where(pred==1.0)[1][0]
            print(classification)
        '''


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



def main():
    cnn = TFConvNN()
    cnn.init_network()
    # cnn.learn()
    # cnn.evaluate_video()

if __name__ == "__main__":
    # main()
    test()
