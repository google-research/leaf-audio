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
"""Creates a flexible learnable frontend.

The class Leaf is a keras layer that can be initialized to replicate
mel-filterbanks, and then be learned via backpropagation.

PreempInit, GaborInit and LowpassInit create keras initializer functions for,
respectively, the pre-emphasis layer, the main convolution layer, and the
lowpass filter.
"""

import gin
from leaf_audio import pooling
from leaf_audio import postprocessing
import tensorflow.compat.v2 as tf
import tensorflow_addons as tfa


gin.external_configurable(tf.keras.regularizers.l1_l2,
                          module='tf.keras.regularizers')


@gin.configurable
def log_compression(inputs: tf.Tensor,
                    log_offset: float = 0.01) -> tf.Tensor:
  """Compress an inputs tensor with using a logarithm."""
  return tf.math.log(inputs + log_offset)


@gin.configurable
class SquaredModulus(tf.keras.layers.Layer):
  """Squared modulus layer.

  Returns a keras layer that implements a squared modulus operator.
  To implement the squared modulus of C complex-valued channels, the expected
  input dimension is N*1*W*(2*C) where channels role alternates between
  real and imaginary part.
  The way the squared modulus is computed is real ** 2 + imag ** 2 as follows:
  - squared operator on real and imag
  - average pooling to compute (real ** 2 + imag ** 2) / 2
  - multiply by 2

  Attributes:
    pool: average-pooling function over the channel dimensions
  """

  def __init__(self):
    super().__init__(name='squared_modulus')
    self._pool = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)

  def call(self, x):
    x = tf.transpose(x, perm=[0, 2, 1])
    output = 2 * self._pool(x**2)
    return tf.transpose(output, perm=[0, 2, 1])


@gin.configurable
class Leaf(tf.keras.models.Model):
  """Keras layer that implements time-domain filterbanks.

  Creates time-domain filterbanks, a learnable front-end that takes an audio
  waveform as input and outputs a learnable spectral representation. This layer
  can be initialized to replicate the computation of standard mel-filterbanks.
  A detailed technical description is presented in Section 2 of
  https://openreview.net/forum?id=jM76BCb6F9m.

  """

  def __init__(self,
               learn_pooling: bool = True,
               learn_filters: bool = True,
               conv1d_cls=tf.keras.layers.Conv1D,
               activation=SquaredModulus(),
               pooling_cls=pooling.LearnablePooling1D,
               n_filters: int = 40,
               sample_rate: int = 16000,
               window_len: float = 25.,
               window_stride: float = 10.,
               compression_fn=tf.identity,
               preemp: bool = False,
               preemp_init: str = 'glorot_uniform',
               complex_conv_init: str = 'glorot_uniform',
               pooling_init: str = 'glorot_uniform',
               regularizer_fn=None,
               mean_var_norm: bool = False,
               spec_augment: bool = False):
    super().__init__(name='tfbanks')
    window_size = int(sample_rate * window_len // 1000 + 1)
    window_stride = int(sample_rate * window_stride // 1000)
    if preemp:
      self._preemp_conv = tf.keras.layers.Conv1D(
          filters=1,
          kernel_size=2,
          strides=1,
          padding='SAME',
          use_bias=False,
          input_shape=(None, None, 1),
          kernel_initializer=preemp_init,
          kernel_regularizer=regularizer_fn if learn_filters else None,
          name='tfbanks_preemp',
          trainable=learn_filters)

    self._complex_conv = conv1d_cls(
        filters=2 * n_filters,
        kernel_size=window_size,
        strides=1,
        padding='SAME',
        use_bias=False,
        input_shape=(None, None, 1),
        kernel_initializer=complex_conv_init,
        kernel_regularizer=regularizer_fn if learn_filters else None,
        name='tfbanks_complex_conv',
        trainable=learn_filters)

    self._activation = activation
    self._pooling = pooling_cls(
        kernel_size=window_size,
        strides=window_stride,
        padding='SAME',
        use_bias=False,
        kernel_initializer=pooling_init,
        kernel_regularizer=regularizer_fn if learn_pooling else None,
        trainable=learn_pooling)

    self._instance_norm = None
    if mean_var_norm:
      self._instance_norm = tfa.layers.InstanceNormalization(
          axis=2,
          epsilon=1e-6,
          center=True,
          scale=True,
          beta_initializer='zeros',
          gamma_initializer='ones',
          name='tfbanks_instancenorm')

    self._compress_fn = compression_fn if compression_fn else tf.identity
    self._spec_augment_fn = postprocessing.SpecAugment(
    ) if spec_augment else tf.identity

    self._preemp = preemp

  def call(self, inputs: tf.Tensor, training: bool = False):
    # Inputs should be [B, W] or [B, W, C]
    outputs = inputs[:, :, tf.newaxis] if inputs.shape.ndims < 3 else inputs
    if self._preemp:
      outputs = self._preemp_conv(outputs)
    outputs = self._complex_conv(outputs)
    outputs = self._activation(outputs)
    outputs = self._pooling(outputs)
    outputs = tf.maximum(outputs, 1e-5)
    outputs = self._compress_fn(outputs)
    if self._instance_norm is not None:
      outputs = self._instance_norm(outputs)
    if training:
      outputs = self._spec_augment_fn(outputs)
    return outputs
