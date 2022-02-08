# coding=utf-8
# Copyright 2022 Google LLC.
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
"""Convolution layers for feature extraction."""

import math
from typing import Tuple

import gin
from leaf_audio import impulse_responses
from leaf_audio import utils
import tensorflow.compat.v2 as tf


def upper_power_of_2(length: tf.Tensor):
  length = tf.cast(length, dtype=tf.float32)
  new_length = 2**tf.math.ceil(tf.math.log(length) / tf.math.log(2.))
  return tf.cast(new_length, dtype=tf.int32)


@gin.configurable
def fft_conv1d(inputs: tf.Tensor, filters: tf.Tensor) -> Tuple[tf.Tensor]:
  """FFT based convolution in 1D.

  We round the input length to the closest upper power of 2 before FFT.

  Args:
   inputs: a tf.Tensor<float>[batch_size, seq_length, 1] of input sequences.
   filters: a tf.Tensor<float>[filter_length, 1, channels]

  Returns:
   A tf.Tensor<float>[batch_size, seq_length, channels] containing the response
   to the 1D convolutions with the filters, and a tf.Tensor<float>[1,] with the
   L1 norm of the filters FFT.
  """
  seq_length = tf.shape(inputs)[1]
  fft_length = upper_power_of_2(seq_length)
  filter_length = tf.shape(filters)[0]
  f_inputs = tf.signal.rfft(
      tf.transpose(inputs, (0, 2, 1)), fft_length=[fft_length])
  f_filters = tf.signal.rfft(
      tf.transpose(filters, (1, 2, 0)), fft_length=[fft_length])
  f_filters_l1 = tf.reduce_sum(tf.math.abs(f_filters))
  result = tf.transpose(
      tf.signal.irfft(f_inputs * tf.math.conj(f_filters)), (0, 2, 1))
  output = tf.roll(result, filter_length // 2 - 1, axis=1)
  output = output[:, :seq_length, :]

  shape = tf.concat([tf.shape(inputs)[:-1], tf.shape(filters)[-1:]], axis=0)
  return tf.reshape(output, shape), f_filters_l1


@gin.configurable
def overlap_add_conv1d(inputs: tf.Tensor,
                       filters: tf.Tensor,
                       fft_length: int = 4096) -> tf.Tensor:
  """FFT based convolution in 1D, using the overlap-add method.

  Args:
   inputs: a tf.Tensor<float>[batch_size, seq_length, 1] of input sequences.
   filters: a tf.Tensor<float>[filter_length, 1, channels].
   fft_length: an int, the length of the Fourier transform.

  Returns:
   A tf.Tensor<float>[batch_size, seq_length, channels] containing the response
   to the 1D convolutions with the filters.
  """
  seq_len = tf.shape(inputs)[1]
  filter_len = tf.shape(filters)[0]
  overlap = filter_len - 1
  seg_size = fft_length - overlap
  f_filters = tf.expand_dims(
      tf.signal.rfft(tf.transpose(filters, (1, 2, 0)), fft_length=[fft_length]),
      axis=2)
  framed = tf.signal.frame(
      tf.transpose(inputs, (0, 2, 1)),
      frame_length=seg_size,
      frame_step=seg_size,
      pad_end=True)
  paddings = [[0, 0], [0, 0], [0, 0], [int(overlap / 2), int(overlap / 2)]]
  framed = tf.pad(framed, paddings)
  f_inputs = tf.signal.rfft(framed, fft_length=[fft_length])
  result = tf.signal.irfft(f_inputs * tf.math.conj(f_filters))
  result = tf.roll(result, filter_len // 2 - 1, axis=-1)
  result = tf.signal.overlap_and_add(result, frame_step=seg_size)
  result = tf.transpose(result, (0, 2, 1))[..., :seq_len, :]
  output = tf.roll(result, 1 - filter_len // 2, axis=1)
  shape = tf.concat([tf.shape(inputs)[:-1], tf.shape(filters)[-1:]], axis=0)
  return tf.reshape(output, shape)


@gin.configurable
class FFTConv1D(tf.keras.layers.Conv1D):
  """Conv1D layer with convolution in the Fourier domain.

     Faster than standard Conv1D for large kernels and low stride.
     Only implemented in "same" padding, and NHWC format for now.
  """

  def call(self, inputs):
    if self.padding == 'causal':
      inputs = tf.pad(inputs, self._compute_causal_padding())

    outputs, frequency_l1 = fft_conv1d(inputs, self.kernel)

    if self.kernel_regularizer:
      self.add_loss(self.kernel_regularizer.l1 * frequency_l1)

    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')

    if self.activation is not None:
      outputs = self.activation(outputs)

    return outputs


class GaborConstraint(tf.keras.constraints.Constraint):
  """Constraint mu and sigma, in radians.

  Mu is constrained in [0,pi], sigma s.t full-width at half-maximum of the
  gaussian response is in [1,pi/2]. The full-width at half maximum of the
  Gaussian response is 2*sqrt(2*log(2))/sigma . See Section 2.2 of
  https://arxiv.org/pdf/1711.01161.pdf for more details.
  """

  def __init__(self, kernel_size):
    """Initialize kernel size.

    Args:
      kernel_size: the length of the filter, in samples.
    """
    self._kernel_size = kernel_size

  def __call__(self, kernel):
    mu_lower = 0.
    mu_upper = math.pi
    sigma_lower = 4 * math.sqrt(2 * math.log(2)) / math.pi
    sigma_upper = self._kernel_size * math.sqrt(2 * math.log(2)) / math.pi
    clipped_mu = tf.clip_by_value(kernel[:, 0], mu_lower, mu_upper)
    clipped_sigma = tf.clip_by_value(kernel[:, 1], sigma_lower, sigma_upper)
    return tf.stack([clipped_mu, clipped_sigma], axis=1)


@gin.configurable
class GaborConv1D(tf.keras.layers.Layer):
  """Implements a convolution with filters defined as complex Gabor wavelets.

  These filters are parametrized only by their center frequency and
  the full-width at half maximum of their frequency response.
  Thus, for n filters, there are 2*n parameters to learn.
  """

  def __init__(self, filters, kernel_size, strides, padding, use_bias,
               input_shape, kernel_initializer, kernel_regularizer, name,
               trainable, sort_filters=False):
    super().__init__(name=name)
    self._filters = filters // 2
    self._kernel_size = kernel_size
    self._strides = strides
    self._padding = padding
    self._use_bias = use_bias
    self._sort_filters = sort_filters
    # Weights are the concatenation of center freqs and inverse bandwidths.
    self._kernel = self.add_weight(
        name='kernel',
        shape=(self._filters, 2),
        initializer=kernel_initializer,
        regularizer=kernel_regularizer,
        trainable=trainable,
        constraint=GaborConstraint(self._kernel_size))
    if self._use_bias:
      self._bias = self.add_weight(name='bias', shape=(self._filters * 2,))

  def call(self, inputs):
    kernel = self._kernel.constraint(self._kernel)
    if self._sort_filters:
      filter_order = tf.argsort(kernel[:, 0])
      kernel = tf.gather(kernel, filter_order, axis=0)
    filters = impulse_responses.gabor_filters(kernel, self._kernel_size)
    real_filters = tf.math.real(filters)
    img_filters = tf.math.imag(filters)
    stacked_filters = tf.stack([real_filters, img_filters], axis=1)
    stacked_filters = tf.reshape(stacked_filters,
                                 [2 * self._filters, self._kernel_size])
    stacked_filters = tf.expand_dims(
        tf.transpose(stacked_filters, perm=(1, 0)), axis=1)
    outputs = tf.nn.conv1d(
        inputs, stacked_filters, stride=self._strides, padding=self._padding)
    if self._use_bias:
      outputs = tf.nn.bias_add(outputs, self._bias, data_format='NWC')
    return outputs


@gin.configurable
class NormalizedConv1D(tf.keras.layers.Layer):
  """A Conv1D which kernel is forced to have L2 norm of 1."""

  def __init__(self,
               filters: int,
               kernel_size: int,
               strides: int,
               padding: int,
               use_bias: bool,
               input_shape: Tuple[int],
               kernel_initializer,
               kernel_regularizer,
               name: str,
               trainable):
    super().__init__(name=name)
    self._kernel = self.add_weight(
        name='kernel',
        shape=(kernel_size, 1, filters),
        initializer=kernel_initializer,
        regularizer=kernel_regularizer,
        trainable=trainable,
        constraint=tf.keras.constraints.UnitNorm(axis=0))
    self._bias = None
    if use_bias:
      self._bias = self.add_weight(name='bias', shape=(filters,))
    self._strides = strides
    self._padding = padding

  def call(self, inputs):
    kernel = self._kernel.constraint(self._kernel)
    outputs = tf.nn.conv1d(
        inputs, kernel, self._strides, self._padding, data_format='NWC')
    if self._bias is not None:
      outputs += self._bias
    return outputs


@gin.configurable
class SincConv1D(tf.keras.layers.Layer):
  """Implements a convolution with filters defined as sinc filters.

  These bandpass filters are parametrized only by their lower cutoff frequency
  and upper cutoff frequency.
  Thus, for n filters, there are 2*n parameters to learn.
  """

  def __init__(self,
               filters: int,
               kernel_size: int,
               strides: int,
               padding: int,
               use_bias: bool,
               input_shape: Tuple[int],
               kernel_initializer,
               kernel_regularizer,
               name: str,
               trainable: bool,
               sample_rate: int = 16000,
               min_low_hz: float = 50.,
               min_band_hz: float = 50.):
    super().__init__(name=name)
    self._filters = filters // 2
    self._kernel_size = kernel_size
    self._strides = strides
    self._padding = padding
    self._use_bias = use_bias
    # Weights are the concatenation of lower and higher cutoff frequencies.
    self._sample_rate = sample_rate
    self._min_low_hz = min_low_hz
    self._min_band_hz = min_band_hz
    self._kernel = self.add_weight(
        name='kernel',
        shape=(self._filters, 2),
        initializer=kernel_initializer,
        regularizer=kernel_regularizer,
        trainable=trainable,
    )
    self._window = utils.window(utils.WindowType.HAMMING, tmax=kernel_size)
    if self._use_bias:
      self._bias = self.add_weight(name='bias', shape=(self._filters,))

  def __call__(self, inputs):
    left_edge = self._min_low_hz + tf.abs(self._kernel[:, 0])
    right_edge = tf.clip_by_value(
        left_edge + self._min_band_hz + tf.abs(self._kernel[:, 1]),
        clip_value_min=self._min_low_hz,
        clip_value_max=self._sample_rate / 2)
    filters = impulse_responses.sinc_filters(
        left_edge[tf.newaxis, :], right_edge[tf.newaxis, :],
        self._kernel_size) * self._window[:, tf.newaxis]
    filters = tf.expand_dims(filters, axis=1)
    outputs = tf.nn.conv1d(
        inputs, filters, stride=self._strides, padding=self._padding)
    if self._use_bias:
      outputs = tf.nn.bias_add(outputs, self._bias, data_format='NWC')
    return outputs
