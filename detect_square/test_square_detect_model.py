from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import tensorflow as tf
import random
from numpy import linalg as LA

def create_square():
  img = np.zeros((28,28),np.uint8)
  top_left = (int(random.random()*26),int(random.random()*26))
  bottom_right = ( int((28 - top_left[0])*random.random()),int((28 - top_left[1])*random.random()))
  #rotate_deg = random.random()*180
  cv2.rectangle(img,top_left,bottom_right,255)
  
  img = img-128.0
  img = (img/128.0)
  #print(img)
  return img.reshape(1,784)

def create_rand():
  img = (np.random.rand(28,28)*255).astype('uint8')
  img = img - 128.0
  img = (img/128.0)
  return img.reshape(1,784)



NUM_LABELS = 2

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, NUM_LABELS])+np.random.rand(784,2) )
b = tf.Variable(tf.zeros([NUM_LABELS])+np.random.rand(2))
y = tf.matmul(x, W) + b

saver = tf.train.Saver()


with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "./model.ckpt")
  print("Model restored.")
  # Do some work with the mod
  print( sess.run(b) )
  print( sess.run(W) )
  
  sum = np.zeros((1,2),np.int64)
  for _ in range(100):
      myY = sess.run(y,feed_dict={x:create_square()})
      sum += sess.run(tf.nn.softmax(myY)).astype(np.int64)
  for _ in range(100):
      myY = sess.run(y,feed_dict={x:create_rand()})
      ret= sess.run(tf.nn.softmax(myY))
      print(ret)
      sum += ret.astype(np.int64)
  print("Sum is {}".format(sum))
