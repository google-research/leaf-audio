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

"""Neural architectures for audio classification."""

import functools
from typing import Any, Optional, Sequence, Tuple

import gin
import tensorflow as tf


@gin.configurable
class AudioClassifier(tf.keras.Model):
  """Neural network architecture to train an audio classifier from waveforms."""

  def __init__(self,
               num_outputs: int,
               frontend: Optional[tf.keras.Model] = None,
               encoder: Optional[tf.keras.Model] = None):
    """Initialization.

    Args:
      num_outputs: the number of classes of the classification problem.
      frontend: A keras model that takes a waveform and outputs a time-frequency
        representation.
      encoder: An encoder to turn the time-frequency representation into an
        embedding.
    """
    super().__init__()
    self._frontend = frontend
    self._encoder = encoder
    self._pool = tf.keras.Sequential([
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Flatten(),
    ])
    self._head = tf.keras.layers.Dense(num_outputs, activation=None)

  def call(self, inputs: tf.Tensor, training: bool = True):
    output = inputs
    if self._frontend is not None:
      output = self._frontend(output, training=training)  # pylint: disable=not-callable
      output = tf.expand_dims(output, -1)
    if self._encoder:
      output = self._encoder(output, training=training)
    output = self._pool(output)
    return self._head(output)


@gin.configurable
class ConvBlock(tf.keras.Sequential):
  """A convolutional block made of two convolutions: time and frequency."""

  def __init__(self,
               num_filters: int,
               activation: Any = 'relu',
               normalization_layer=tf.keras.layers.BatchNormalization,
               dropout: float = 0.0,
               max_pooling: bool = True):
    conv_fn = functools.partial(tf.keras.layers.Conv2D, padding='SAME')
    layers = [
        conv_fn(filters=num_filters, kernel_size=[3, 1]),
        tf.keras.layers.Activation(activation)
    ]
    if normalization_layer is not None:
      layers.append(normalization_layer())
    if dropout > 0.0:
      layers.append(tf.keras.layers.Dropout(dropout))
    layers.extend([
        conv_fn(filters=num_filters, kernel_size=[1, 3]),
        tf.keras.layers.Activation(activation)
    ])
    if normalization_layer is not None:
      layers.append(normalization_layer())
    if dropout > 0.0:
      layers.append(tf.keras.layers.Dropout(dropout))
    if max_pooling:
      layers.append(tf.keras.layers.MaxPool2D())
    super().__init__(layers)


@gin.configurable
class ConvNet(tf.keras.Sequential):
  """An convolutional encoder made of a sequence of ConvBlocks."""

  def __init__(self, filters: Sequence[int] = gin.REQUIRED, **kwargs):
    # Uses pooling every second layer
    super().__init__([ConvBlock(depth, max_pooling=(not i or i % 2), **kwargs)
                      for (i, depth) in enumerate(filters)])


@gin.configurable
class PANNWavBlock(tf.keras.Sequential):
  """Frontend convolutional block for PANN's WaveGram."""

  def __init__(self, filters: int, pool_size: int):
    """Initializes PANNWavBlock.

    Args:
      filters: the number of filters.
      pool_size: kernel size and stride of the 1D max-pooling.
    """
    super().__init__()

    for dilation_rate in [1, 2]:
      self.add(tf.keras.layers.Conv1D(
          filters=filters,
          kernel_size=3,
          strides=1,
          dilation_rate=dilation_rate,
          padding='SAME',
          use_bias=False,))
      self.add(tf.keras.layers.BatchNormalization())
      self.add(tf.keras.layers.ReLU())

    self.add(tf.keras.layers.MaxPool1D(
        pool_size=pool_size, strides=pool_size))


@gin.configurable
class PANNConvBlock(tf.keras.Sequential):
  """PANNConvBlock for PANN's main architecture."""

  def __init__(self, filters: int, pool_size: Tuple[int, int]):
    """Initializes PANNConvBlock.

    Args:
      filters: the number of filters.
      pool_size: kernel size (also used as stride) of the 2D max-pooling.
    """
    super().__init__()
    for _ in range(2):
      self.add(
          tf.keras.layers.Conv2D(
              filters=filters,
              kernel_size=(3, 3),
              strides=(1, 1),
              padding='SAME',
              use_bias=False,
          ))
      self.add(tf.keras.layers.BatchNormalization())
      self.add(tf.keras.layers.ReLU())

    self.add(
        tf.keras.layers.AvgPool2D(
            pool_size=pool_size, strides=pool_size, padding='SAME'))


@gin.configurable
class WaveGram(tf.keras.Model):
  """WaveGram frontend from PANN (https://arxiv.org/abs/1912.10211)."""

  def __init__(self):

    super().__init__()

    self.pre_conv0 = tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=11,
        strides=5,
        padding='SAME',
        use_bias=False,
    )
    self.pre_bn0 = tf.keras.layers.BatchNormalization()
    self.pre_blocks = tf.keras.Sequential(
        [PANNWavBlock(filters=dim, pool_size=4) for dim in [64, 128, 128]])
    self.last_block = PANNConvBlock(filters=64, pool_size=(2, 1))

  def call(self, inputs, training):
    outputs = inputs[:, :, tf.newaxis] if inputs.shape.ndims < 3 else inputs
    outputs = tf.nn.relu(
        self.pre_bn0(self.pre_conv0(outputs), training=training))
    outputs = self.pre_blocks(outputs, training)
    outputs = outputs[..., tf.newaxis, :]
    outputs = self.last_block(outputs, training=training)
    return tf.transpose(outputs, (0, 1, 3, 2))


@gin.configurable
class PANN(tf.keras.Sequential):
  """CNNX from PANN (https://arxiv.org/abs/1912.10211). Default is CNN14."""

  def __init__(self, depth: int = 6, dropout_rate: float = 0.2):
    """Initializes PANN model.

    Args:
      depth: the number of convolutional blocks. First block has 64 filters and
        the number of filters doubles at every block up to 2048 filters. A pool
        size of (2, 2) is used for the first 5 blocks and then (1, 1) to avoid
        negative output size.
      dropout_rate: the dropout rate.
    """
    super().__init__(name='pann')
    self.add(tf.keras.layers.BatchNormalization(axis=2))
    for block_idx in range(depth):
      filters = min(64 * (2**block_idx), 2048)
      pool_size = (2, 2) if block_idx < 5 else (1, 1)
      self.add(PANNConvBlock(
          filters=filters,
          pool_size=pool_size,
      ))
      self.add(tf.keras.layers.Dropout(dropout_rate))
