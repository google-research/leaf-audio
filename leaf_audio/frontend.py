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

import functools
from typing import Callable, Optional

import gin
from leaf_audio import convolution
from leaf_audio import initializers
from leaf_audio import pooling
from leaf_audio import postprocessing
from leaf_audio import utils
import tensorflow.compat.v2 as tf
import tensorflow_addons as tfa


_TensorCallable = Callable[[tf.Tensor], tf.Tensor]
_Initializer = tf.keras.initializers.Initializer

gin.external_configurable(tf.keras.regularizers.l1_l2,
                          module='tf.keras.regularizers')


@gin.configurable
def log_compression(inputs: tf.Tensor,
                    log_offset: float = 1e-5) -> tf.Tensor:
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

  Creates a LEAF frontend, a learnable front-end that takes an audio
  waveform as input and outputs a learnable spectral representation. This layer
  can be initialized to replicate the computation of standard mel-filterbanks.
  A detailed technical description is presented in Section 3 of
  https://arxiv.org/abs/2101.08596 .

  """

  def __init__(
      self,
      learn_pooling: bool = True,
      learn_filters: bool = True,
      conv1d_cls=convolution.GaborConv1D,
      activation=SquaredModulus(),
      pooling_cls=pooling.GaussianLowpass,
      n_filters: int = 40,
      sample_rate: int = 16000,
      window_len: float = 25.,
      window_stride: float = 10.,
      compression_fn: _TensorCallable = postprocessing.PCENLayer(
          alpha=0.96,
          smooth_coef=0.04,
          delta=2.0,
          floor=1e-12,
          trainable=True,
          learn_smooth_coef=True,
          per_channel_smooth_coef=True),
      preemp: bool = False,
      preemp_init: _Initializer = initializers.PreempInit(),
      complex_conv_init: _Initializer = initializers.GaborInit(
          sample_rate=16000, min_freq=60.0, max_freq=7800.0),
      pooling_init: _Initializer = tf.keras.initializers.Constant(0.4),
      regularizer_fn: Optional[tf.keras.regularizers.Regularizer] = None,
      mean_var_norm: bool = False,
      spec_augment: bool = False,
      name='leaf'):
    super().__init__(name=name)
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

  def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
    """Computes the Leaf representation of a batch of waveforms.

    Args:
      inputs: input audio of shape (batch_size, num_samples) or (batch_size,
        num_samples, 1).
      training: training mode, controls whether SpecAugment is applied or not.

    Returns:
      Leaf features of shape (batch_size, time_frames, freq_bins).
    """
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


@gin.configurable
class TimeDomainFilterbanks(Leaf):
  """Time-Domain Filterbanks frontend.

  See Section 2 of https://arxiv.org/abs/1711.01161 for reference.
  """

  def __init__(self, sample_rate=16000, name='tfbanks', **kwargs):
    """Constructor of a SincNet + frontend.


    Args:
      sample_rate: audio sampling rate.
      name: name of the layer.
      **kwargs: Arguments passed to Leaf, except conv1d_cls, complex_conv_init,
        activation, pooling_cls, pooling_init, compression_fn,
        sample_rate and name which are already fixed.
    """
    complex_conv_init = initializers.GaborInit(
        sample_rate=sample_rate,
        min_freq=60.0,
        max_freq=7800.0)
    pooling_init = initializers.LowpassInit(
        sample_rate=sample_rate, window_type=utils.WindowType.SQUARED_HANNING)
    super().__init__(
        conv1d_cls=tf.keras.layers.Conv1D,
        activation=SquaredModulus(),
        pooling_cls=pooling.LearnablePooling1D,
        complex_conv_init=complex_conv_init,
        pooling_init=pooling_init,
        compression_fn=functools.partial(log_compression, log_offset=1e-5),
        name=name,
        **kwargs)


@gin.configurable
class SincNet(Leaf):
  """SincNet frontend.

  See Section 2 of https://arxiv.org/abs/1808.00158 for reference.
  """

  def __init__(self, name='sincnet', **kwargs):
    """Constructor of a SincNet frontend.

    Args:
      name: name of the layer.
      **kwargs: Arguments passed to Leaf, except conv1d_cls, complex_conv_init,
        activation, pooling_cls, compression_fn and name which are already
        fixed.
    """

    super().__init__(conv1d_cls=convolution.SincConv1D,
                     complex_conv_init=initializers.SincInit(),
                     activation=tf.keras.layers.LeakyReLU(alpha=0.2),
                     pooling_cls=pooling.MaxPooling1D,
                     compression_fn=tf.keras.layers.LayerNormalization(),
                     name=name,
                     **kwargs)


@gin.configurable
class SincNetPlus(Leaf):
  """SincNet+ frontend.

  It replaces max-pooling with a Gaussian lowpass, and LayerNorm with PCEN.
  """

  def __init__(self, name='sincnet_plus', **kwargs):
    """Constructor of a SincNet + frontend.


    Args:
      name: name of the layer.
      **kwargs: Arguments passed to Leaf, except conv1d_cls, complex_conv_init,
        activation, pooling_cls, pooling_init, compression_fn and name which are
        already fixed.
    """
    super().__init__(
        conv1d_cls=convolution.SincConv1D,
        complex_conv_init=initializers.SincInit(),
        activation=tf.keras.layers.LeakyReLU(alpha=0.2),
        pooling_cls=pooling.GaussianLowpass,
        pooling_init=tf.keras.initializers.Constant(0.4),
        compression_fn=postprocessing.PCENLayer(),
        name=name,
        **kwargs)


@gin.configurable
class MelFilterbanks(tf.keras.layers.Layer):
  """Computes mel-filterbanks."""

  def __init__(self,
               n_filters: int = 40,
               sample_rate: int = 16000,
               n_fft: int = 512,
               window_len: float = 25.,
               window_stride: float = 10.,
               compression_fn: _TensorCallable = log_compression,
               min_freq: float = 60.0,
               max_freq: float = 7800.0,
               **kwargs):
    """Constructor of a MelFilterbanks frontend.

    Args:
      n_filters: the number of mel_filters.
      sample_rate: sampling rate of input waveforms, in samples.
      n_fft: number of frequency bins of the spectrogram.
      window_len: size of the window, in seconds.
      window_stride: stride of the window, in seconds.
      compression_fn: a callable, the compression function to use.
      min_freq: minimum frequency spanned by mel-filters (in Hz).
      max_freq: maximum frequency spanned by mel-filters (in Hz).
      **kwargs: other arguments passed to the base class, e.g. name.
    """

    super().__init__(**kwargs)

    self._n_filters = n_filters
    self._sample_rate = sample_rate
    self._n_fft = n_fft
    self._window_len = int(sample_rate * window_len // 1000 + 1)
    self._window_stride = int(sample_rate * window_stride // 1000)
    self._compression_fn = compression_fn
    self._min_freq = min_freq
    self._max_freq = max_freq if max_freq else sample_rate / 2.

    self.mel_filters = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=self._n_filters,
        num_spectrogram_bins=self._n_fft // 2 + 1,
        sample_rate=self._sample_rate,
        lower_edge_hertz=self._min_freq,
        upper_edge_hertz=self._max_freq)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    """Computes mel-filterbanks of a batch of waveforms.

    Args:
      inputs: input audio of shape (batch_size, num_samples).

    Returns:
      Mel-filterbanks of shape (batch_size, time_frames, freq_bins).
    """
    stft = tf.signal.stft(
        inputs,
        frame_length=self._window_len,
        frame_step=self._window_stride,
        fft_length=self._n_fft,
        pad_end=True)

    spectrogram = tf.math.square(tf.math.abs(stft))

    mel_filterbanks = tf.matmul(spectrogram, self.mel_filters)
    mel_filterbanks = self._compression_fn(mel_filterbanks)
    return mel_filterbanks
