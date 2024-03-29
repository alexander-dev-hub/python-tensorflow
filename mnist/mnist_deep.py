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



"""A deep MNIST classifier using convolutional layers.



See extensive documentation at

https://www.tensorflow.org/get_started/mnist/pros

"""

# Disable linter warnings to maintain consistency with tutorial.

# pylint: disable=invalid-name

# pylint: disable=g-bad-import-order



from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import argparse

import sys

import tempfile



from tensorflow.examples.tutorials.mnist import input_data



import tensorflow as tf



import numpy



FLAGS = None





def deepnn(x):

  """deepnn builds the graph for a deep net for classifying digits.



  Args:

    x: an input tensor with the dimensions (N_examples, 784), where 784 is the

    number of pixels in a standard MNIST image.



  Returns:

    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values

    equal to the logits of classifying the digit into one of 10 classes (the

    digits 0-9). keep_prob is a scalar placeholder for the probability of

    dropout.

  """

  # Reshape to use within a convolutional neural net.

  # Last dimension is for "features" - there is only one here, since images are

  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.

  with tf.name_scope('reshape'):
    '''
    x를 4D 텐서로 reshape해야 합니다. 두 번째와 세 번째 차원은 이미지의 가로와 세로 길이, 그리고 마지막 차원은 컬러 채널의 수를 나타냅니다
    '''
    x_image = tf.reshape(x, [-1, 28, 28, 1])



  # First convolutional layer - maps one grayscale image to 32 feature maps.
    '''
    합성곱 계층에서는 5x5의 윈도우(patch라고도 함) 크기를 가지는 32개의 필터를 사용하며,
    따라서 구조(shape)가 [5, 5, 1, 32]인 가중치 텐서를 정의해야 합니다. 
    처음 두 개의 차원은 윈도우의 크기, 세 번째는 입력 채널의 수, 마지막은 출력 채널의 수(즉, 얼마나 많은 특징을 사용할 것인가)를 나타냅니다. 
    또한, 각각의 출력 채널에 대한 편향을 정의해야 합니다.
    '''

  with tf.name_scope('conv1'):

    W_conv1 = weight_variable([5, 5, 1, 32])

    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)



  # Pooling layer - downsamples by 2X.

  with tf.name_scope('pool1'):

    h_pool1 = max_pool_2x2(h_conv1)



  # Second convolutional layer -- maps 32 feature maps to 64.
    '''
    심층 신경망을 구성하기 위해서, 앞에서 만든 것과 비슷한 계층을 쌓아올릴 수 있습니다. 여기서는 두 번째 합성곱 계층이 5x5 윈도우에 64개의 필터를 가집니다.
    '''
  with tf.name_scope('conv2'):

    W_conv2 = weight_variable([5, 5, 32, 64])

    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)



  # Second pooling layer.

  with tf.name_scope('pool2'):

    h_pool2 = max_pool_2x2(h_conv2)



  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image

  # is down to 7x7x64 feature maps -- maps this to 1024 features.
    '''
    두 번째 계층을 거친 뒤 이미지 크기는 7x7로 줄어들었습니다. 이제 여기에 1024개의 뉴런으로 연결되는 완전 연결 계층을 구성합니다.
    이를 위해서 7x7 이미지의 배열을 reshape해야 하며, 완전 연결 계층에 맞는 가중치 행렬과 편향 행렬을 구성합니다. 
    최종적으로 완전 연결 계층의 끝에 ReLU 함수를 적용합니다.
    '''
  with tf.name_scope('fc1'):

    W_fc1 = weight_variable([7 * 7 * 64, 1024])

    b_fc1 = bias_variable([1024])



    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)



  # Dropout - controls the complexity of the model, prevents co-adaptation of

  # features.
    '''
    오버피팅(overfitting) 되는 것을 방지하기 위해, 드롭아웃을 적용할 것입니다. 뉴런이 드롭아웃되지 않을 확률을 저장하는 placeholder를 만듭니다. 
    이렇게 하면 나중에 드롭아웃이 훈련 과정에는 적용되고, 테스트 과정에서는 적용되지 않도록 설정할 수 있습니다.
    TensorFlow의 tf.nn.dropout 함수는 뉴런의 출력을 자동으로 스케일링(scaling)하므로, 추가로 스케일링 할 필요 없이 그냥 드롭아웃을 적용할 수 있습니다
    '''
  with tf.name_scope('dropout'):

    keep_prob = tf.placeholder(tf.float32)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



  # Map the 1024 features to 10 classes, one for each digit
    '''
    마지막으로, 위에서 단일 계층 소프트맥스 회귀 모델을 구성할 때와 비슷하게 아래 코드와 같이 소프트맥스 계층을 추가합니다.
    '''
  with tf.name_scope('fc2'):

    W_fc2 = weight_variable([1024, 10])

    b_fc2 = bias_variable([10])



    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  return y_conv, keep_prob





def conv2d(x, W):

  """conv2d returns a 2d convolution layer with full stride.
  스트라이드를 1로, 출력 크기가 입력과 같게 되도록 0으로 패딩하도록 설정합니다.
  """

  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')





def max_pool_2x2(x):

  """max_pool_2x2 downsamples a feature map by 2X.
  풀링은 2x2 크기의 맥스 풀링을 적용합니다.
    """

  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],

                        strides=[1, 2, 2, 1], padding='SAME')





def weight_variable(shape):

  """weight_variable generates a weight variable of a given shape.
   대칭성을 깨뜨리고 기울기(gradient)가 0이 되는 것을 방지하기 위해, 가중치에 약간의 잡음을 주어 초기화합니다. 
   또한, 모델에 ReLU 뉴런이 포함되므로, "죽은 뉴런"을 방지하기 위해 편향을 작은 양수(0.1)로 초기화합니다.
  
  """

  initial = tf.truncated_normal(shape, stddev=0.1)

  return tf.Variable(initial)





def bias_variable(shape):

  """bias_variable generates a bias variable of a given shape."""

  initial = tf.constant(0.1, shape=shape)

  return tf.Variable(initial)





def main(_):

  # Import data

  mnist = input_data.read_data_sets(FLAGS.data_dir)



  # Create the model

  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer

  y_ = tf.placeholder(tf.int64, [None])



  # Build the graph for the deep net

  y_conv, keep_prob = deepnn(x)



  with tf.name_scope('loss'):

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(

        labels=y_, logits=y_conv)

  cross_entropy = tf.reduce_mean(cross_entropy)



  with tf.name_scope('adam_optimizer'):

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)



  with tf.name_scope('accuracy'):

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)

    correct_prediction = tf.cast(correct_prediction, tf.float32)

  accuracy = tf.reduce_mean(correct_prediction)



  graph_location = tempfile.mkdtemp()

  print('Saving graph to: %s' % graph_location)

  train_writer = tf.summary.FileWriter(graph_location)

  train_writer.add_graph(tf.get_default_graph())



  with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(2000):

      batch = mnist.train.next_batch(50)

      if i % 100 == 0:

        train_accuracy = accuracy.eval(feed_dict={

            x: batch[0], y_: batch[1], keep_prob: 1.0})

        print('step %d, training accuracy %g' % (i, train_accuracy))

      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})



    # compute in batches to avoid OOM on GPUs 

    accuracy_l = []

    for _ in range(20):

      batch = mnist.test.next_batch(500, shuffle=False)

      accuracy_l.append(accuracy.eval(feed_dict={x: batch[0], 

                                                 y_: batch[1], 

                                                 keep_prob: 1.0}))

    print('test accuracy %g' % numpy.mean(accuracy_l))





if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('--data_dir', type=str,

                      default='f:/tmp/mnist/input_data',

                      help='Directory for storing input data')

  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main )