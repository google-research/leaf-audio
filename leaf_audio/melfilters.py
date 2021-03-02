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
"""Computes mel-filterbanks and corresponding gabor filters.

This script creates a class Gabor which takes as arguments parameters for a
mel-filterbank. A family of gabor wavelets is then created to match the
frequency response of the mel-filterbank.
"""

import gin
from leaf_audio import impulse_responses
import numpy as np
import tensorflow as tf


@gin.configurable
class Gabor:
  """This class creates gabor filters designed to match mel-filterbanks.

  Attributes:
    n_filters: number of filters
    min_freq: minimum frequency spanned by the filters
    max_freq: maximum frequency spanned by the filters
    sample_rate: samplerate (samples/s)
    window_len: window length in samples
    n_fft: number of frequency bins to compute mel-filters
    normalize_energy: boolean, True means that all filters have the same energy,
      False means that the higher the center frequency of a filter, the higher
      its energy
  """

  def __init__(self,
               n_filters: int = 40,
               min_freq: float = 0.,
               max_freq: float = 8000.,
               sample_rate: int = 16000,
               window_len: int = 401,
               n_fft: int = 512,
               normalize_energy: bool = False):

    self.n_filters = n_filters
    self.min_freq = min_freq
    self.max_freq = max_freq
    self.sample_rate = sample_rate
    self.window_len = window_len
    self.n_fft = n_fft
    self.normalize_energy = normalize_energy

  @property
  def gabor_params_from_mels(self):
    """Retrieves center frequencies and standard deviations of gabor filters."""
    coeff = tf.math.sqrt(2. * tf.math.log(2.)) * self.n_fft
    sqrt_filters = tf.math.sqrt(self.mel_filters)
    center_frequencies = tf.cast(
        tf.argmax(sqrt_filters, axis=1), dtype=tf.float32)
    peaks = tf.reduce_max(sqrt_filters, axis=1, keepdims=True)
    half_magnitudes = peaks / 2.
    fwhms = tf.reduce_sum(
        tf.cast(sqrt_filters >= half_magnitudes, dtype=tf.float32), axis=1)
    return tf.stack(
        [center_frequencies * 2 * np.pi / self.n_fft, coeff / (np.pi * fwhms)],
        axis=1)

  def _mel_filters_areas(self, filters):
    """Area under each mel-filter."""
    peaks = tf.reduce_max(filters, axis=1, keepdims=True)
    return peaks * (tf.reduce_sum(
        tf.cast(filters > 0, dtype=tf.float32), axis=1, keepdims=True) +
                    2) * np.pi / self.n_fft

  @property
  def mel_filters(self):
    """Creates a bank of mel-filters."""
    # build mel filter matrix
    mel_filters = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=self.n_filters,
        num_spectrogram_bins=self.n_fft // 2 + 1,
        sample_rate=self.sample_rate,
        lower_edge_hertz=self.min_freq,
        upper_edge_hertz=self.max_freq)
    mel_filters = tf.transpose(mel_filters, [1, 0])
    if self.normalize_energy:
      mel_filters = mel_filters / self._mel_filters_areas(mel_filters)
    return mel_filters

  @property
  def gabor_filters(self):
    """Generates gabor filters that match the corresponding mel-filters."""
    gabor_filters = impulse_responses.gabor_filters(
        self.gabor_params_from_mels, size=self.window_len)
    return gabor_filters * tf.cast(
        tf.math.sqrt(
            self._mel_filters_areas(self.mel_filters) * 2 *
            tf.math.sqrt(np.pi) * self.gabor_params_from_mels[:, 1:2]),
        dtype=tf.complex64)
