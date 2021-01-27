# coding=utf-8
# Copyright 2021 Google LLC.
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

# Lint as: python3
"""Convolution layers for pooling."""

import gin
from leaf_audio import impulse_responses
import tensorflow.compat.v2 as tf


@gin.configurable
class LearnablePooling1D(tf.keras.layers.Layer):
  """Learnable pooling in 1D.

  This layer is a learnable pooling, implemented as a convolution layer:
    - There is only one filter, the pooling function
    - The input is reshaped from NWC to NCW1 and the pooling is implemented as
    a 2D filter of size (1, kernel_size), to pool channels one-by-one.
    - The output of this operation is then reshaped to NWC
  """

  def __init__(self,
               kernel_size,
               strides=1,
               padding='same',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               kernel_regularizer=None,
               trainable=False):

    super().__init__(name='learnable_pooling')
    self.pool = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(1, kernel_size),
        strides=(1, strides),
        padding='valid',
        use_bias=False,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        trainable=trainable)

  def call(self, x):
    x = tf.expand_dims(x, axis=1)
    x = tf.transpose(x, perm=[0, 3, 2, 1])
    x = self.pool(x)
    x = tf.transpose(x, perm=[0, 3, 2, 1])
    return tf.squeeze(x, axis=1)


@gin.configurable
class ChannelWiseLearnablePooling1D(tf.keras.layers.Layer):
  """Channel wise learnable pooling in 1D.

  Implemented as a depthwise convolution,
  where each filter is initialized as a lowpass filter and sees one input
  channel.
  """

  def __init__(self,
               kernel_size,
               strides=1,
               padding='same',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               kernel_regularizer=None,
               trainable=False):

    super().__init__(name='learnable_pooling')
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.kernel_regularizer = kernel_regularizer
    self.trainable = trainable

  def build(self, input_shape):
    self.kernel = self.add_weight(
        name='kernel',
        shape=(self.kernel_size, 1, input_shape[2]),
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        trainable=self.trainable)

  def call(self, inputs):
    return tf.nn.conv1d(
        inputs, self.kernel, stride=self.strides, padding=self.padding.upper())


@gin.configurable
class GaussianLowpass(tf.keras.layers.Layer):
  """Depthwise pooling (each input filter has its own pooling filter).

  Pooling filters are parametrized as zero-mean Gaussians, with learnable
  std. They can be initialized with tf.keras.initializers.Constant(0.4)
  to approximate a Hanning window.
  We rely on depthwise_conv2d as there is no depthwise_conv1d in Keras so far.
  """

  def __init__(
      self,
      kernel_size,
      strides=1,
      padding='same',
      use_bias=True,
      kernel_initializer='glorot_uniform',
      kernel_regularizer=None,
      trainable=False,
  ):

    super().__init__(name='learnable_pooling')
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.kernel_regularizer = kernel_regularizer
    self.trainable = trainable

  def build(self, input_shape):
    self.kernel = self.add_weight(
        name='kernel',
        shape=(1, 1, input_shape[2], 1),
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        trainable=self.trainable)

  def call(self, inputs):
    kernel = impulse_responses.gaussian_lowpass(self.kernel, self.kernel_size)
    outputs = tf.expand_dims(inputs, axis=1)
    outputs = tf.nn.depthwise_conv2d(
        outputs,
        kernel,
        strides=(1, self.strides, self.strides, 1),
        padding=self.padding.upper())
    return tf.squeeze(outputs, axis=1)


@gin.configurable
class MaxPooling1D(tf.keras.layers.MaxPool1D):
  """Max pooling in 1D.

  Most parameters are not used, but kept for compatibility with other poolings.
  """

  def __init__(self,
               kernel_size,
               strides=1,
               padding='same',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               kernel_regularizer=None,
               trainable=False):

    super().__init__(
        pool_size=kernel_size,
        strides=strides,
        padding=padding,
        name='frontend_max_pooling')
