# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from os import listdir
from os.path import isfile, join
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cv2
import tensorflow as tf

FLAGS = None

NUM_LABELS = 2
TEST_PERCENT = 0.4

def main(_):
  # Import data
  #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  onlyfiles = [f for f in listdir('./data/') if isfile(join('./data/', f))]
#  onlyfiles = onlyfiles[np.random.permutation(len(onlyfiles)).astype('uint8')]
  train_size = int(len(onlyfiles)*(1-TEST_PERCENT))
  test_size  = int(len(onlyfiles) - train_size)

  train_images = np.zeros((train_size,784),np.float32)
  train_labels = np.zeros((train_size,NUM_LABELS),np.float32) #rect or no rect

  test_images = np.zeros((test_size,784),np.float32)
  test_labels = np.zeros((test_size,NUM_LABELS),np.float32) #rect or no rect



  counter = 0
  for file in onlyfiles:
      img = cv2.imread('./data/'+file)[:,:,0].reshape(784)
      #print(img)
      eq_img = ((img-128.0)/128.0)
      #print(eq_img)

      #print(file)
      if counter < train_size:
          train_images[counter,:] = eq_img
          if 'rect' in file:
              train_labels[counter,0] = 1.0
          else:
              train_labels[counter,1] = 1.0
      else:  
          test_images[counter-train_size,:] = eq_img
          if 'rect' in file:
              test_labels[counter-train_size,0] = 1.0
          else:
              test_labels[counter-train_size,1] = 1.0
      counter += 1
#  print(train_images[0,:])
#  sys.exit(0)
#  print( train_labels)
#  print( test_labels )
#  import sys
#  sys.exit(0)



  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, NUM_LABELS])+np.random.rand(784,2) )
  b = tf.Variable(tf.zeros([NUM_LABELS])+np.random.rand(2))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, NUM_LABELS])

  saver = tf.train.Saver()


  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  batchs = int(len(onlyfiles)/10)
  train_batch_size = int(train_size / batchs)
  test_batch_size = int(test_size / batchs)
  train_offset = 0
  test_offset = 0
  for _ in range(batchs):
    #batch_xs, batch_ys = mnist.train.next_batch(100)
    batch_xs = train_images[train_offset:train_offset+train_batch_size]
    batch_ys = train_labels[train_offset:train_offset+train_batch_size]
    train_offset += train_batch_size

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print(sess.run(y,feed_dict={x:batch_xs}))
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #test_xs = test_images[test_offset:test_offset+test_batch_size]
    #test_ys = test_labels[test_offset:test_offset+test_batch_size]
    #test_offset += test_batch_size
    test_xs = test_images
    test_ys = test_labels
    print(sess.run(accuracy, feed_dict={x: test_xs,
                                      y_: test_ys}))


    print(sess.run(W))
    cv2.imshow('rect_filter',(sess.run(W)[:,0].reshape(28,28)*255).astype('uint8'))
    cv2.imshow('not rect',(sess.run(W)[:,1].reshape(28,28)*255).astype('uint8'))
    cv2.waitKey(10)

  save_path = saver.save(sess, "./model.ckpt")
  print("Model saved in file: %s" % save_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
