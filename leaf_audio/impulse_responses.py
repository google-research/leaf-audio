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

"""Generate impulse responses for several type of filters."""

import math
import tensorflow.compat.v2 as tf


def amplitude(filters: tf.Tensor, i: int) -> float:
  return tf.math.abs(tf.signal.fftshift(tf.signal.fft(filters[:, i])))


def gabor_impulse_response(t: tf.Tensor, center: tf.Tensor,
                           fwhm: tf.Tensor) -> tf.Tensor:
  """Computes the gabor impulse response."""
  denominator = 1.0 / (tf.math.sqrt(2.0 * math.pi) * fwhm)
  gaussian = tf.exp(tf.tensordot(1.0 / (2. * fwhm**2), -t**2, axes=0))
  center_frequency_complex = tf.cast(center, tf.complex64)
  t_complex = tf.cast(t, tf.complex64)
  sinusoid = tf.math.exp(
      1j * tf.tensordot(center_frequency_complex, t_complex, axes=0))
  denominator = tf.cast(denominator, dtype=tf.complex64)[:, tf.newaxis]
  gaussian = tf.cast(gaussian, dtype=tf.complex64)
  return denominator * sinusoid * gaussian


def gabor_filters(kernel, size: int = 401) -> tf.Tensor:
  """Computes the gabor filters from its parameters for a given size.

  Args:
    kernel: tf.Tensor<float>[filters, 2] the parameters of the Gabor kernels.
    size: the size of the output tensor.

  Returns:
    A tf.Tensor<float>[filters, size].
  """
  return gabor_impulse_response(
      tf.range(-(size // 2), (size + 1) // 2, dtype=tf.float32),
      center=kernel[:, 0], fwhm=kernel[:, 1])


def sinc_impulse_response(t: tf.Tensor, frequency: tf.Tensor) -> tf.Tensor:
  """Computes the sinc impulse response."""
  return tf.sin(2*math.pi*frequency*t) / (2*math.pi*frequency*t)


def sinc_filters(cutoff_freq_low: tf.Tensor,
                 cutoff_freq_high: tf.Tensor,
                 size: int = 401,
                 sample_rate: int = 16000) -> tf.Tensor:
  """Computes the sinc filters from its parameters for a given size.

  Sinc is not defined in zero so we need to separately compute negative
  (left_range) and positive part (right_range).

  Args:
    cutoff_freq_low: tf.Tensor<float>[1, filters] the lower cutoff frequencies
      of the bandpass.
    cutoff_freq_high: tf.Tensor<float>[1, filters] the upper cutoff frequencies
      of the bandpass.
    size: the size of the output tensor.
    sample_rate: audio sampling rate

  Returns:
    A tf.Tensor<float>[size, filters].
  """
  left_range = tf.range(
      -(size // 2), 0, dtype=tf.float32)[:, tf.newaxis] / tf.cast(
          sample_rate, dtype=tf.float32)
  right_range = tf.range(
      1, size // 2 + 1, dtype=tf.float32)[:, tf.newaxis] / tf.cast(
          sample_rate, dtype=tf.float32)
  high_pass_left_range = 2 * cutoff_freq_high * sinc_impulse_response(
      left_range, cutoff_freq_high)
  high_pass_right_range = 2 * cutoff_freq_high * sinc_impulse_response(
      right_range, cutoff_freq_high)
  low_pass_left_range = 2 * cutoff_freq_low * sinc_impulse_response(
      left_range, cutoff_freq_low)
  low_pass_right_range = 2 * cutoff_freq_low * sinc_impulse_response(
      right_range, cutoff_freq_low)
  high_pass = tf.concat(
      [high_pass_left_range, 2 * cutoff_freq_high, high_pass_right_range],
      axis=0)
  low_pass = tf.concat(
      [low_pass_left_range, 2 * cutoff_freq_low, low_pass_right_range], axis=0)
  band_pass = high_pass - low_pass
  return band_pass / tf.reduce_max(band_pass, axis=0, keepdims=True)


def gaussian_lowpass(sigma: tf.Tensor, filter_size: int):
  """Generates gaussian windows centered in zero, of std sigma.

  Args:
    sigma: tf.Tensor<float>[1, 1, C, 1] for C filters.
    filter_size: length of the filter.

  Returns:
    A tf.Tensor<float>[1, filter_size, C, 1].
  """
  sigma = tf.clip_by_value(
      sigma, clip_value_min=(2. / filter_size), clip_value_max=0.5)
  t = tf.range(0, filter_size, dtype=tf.float32)
  t = tf.reshape(t, (1, filter_size, 1, 1))
  numerator = t - 0.5 * (filter_size - 1)
  denominator = sigma * 0.5 * (filter_size - 1)
  return tf.math.exp(-0.5 * (numerator / denominator)**2)
