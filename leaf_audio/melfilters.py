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
from leaf_audio import utils
import numpy as np


@gin.configurable
class Gabor:
  """Creates a bank of Gabor wavelets: a bank of band-pass filters that match mel-filterbanks.

  The gabor class takes as arguments the parameters of a mel-filterbank.
  It can then create gabor wavelets that match the frequency response of
  the mel-filterbank.

  Attributes:
    n_filters: number of filters
    min_freq: minimum frequency spanned by the filters
    max_freq: maximum frequency spanned by the filters
    sample_rate: samplerate (samples/s)
    window_len: window size in ms
    window_stride: window stride in ms
    n_fft: number of frequency bins to compute mel-filters
    normalize_energy: boolean, True means that all filters have the same energy,
      False means that the higher the center frequency of a filter, the higher
      its energy
    filt_edge: list of start and end points of mel-filters along the frequency
      axis
    mel_filters: list of mel-filters (each represented as a numpy array)
    center_frequencies: list of center frequency of mel-filters (and
      corresponding gabor filters)
    fwhms: list of full-width at half maximum (FWHM) of mel-filters (and
      corresponding gabor filters)
    gabor_filters: list of gabor wavelets (complex valued numpy arrays)
  Methods:
    _build_mels: creates a bank of mel-filters
    _build_gabors: creates a bank of gabor filters that matches the mel-filters
  """

  def __init__(self,
               n_filters: int = 40,
               min_freq: float = 0.,
               max_freq: float = 8000.,
               sample_rate: int = 16000,
               window_len: float = 25.,
               window_stride: float = 10.,
               n_fft: int = 512,
               normalize_energy: bool = True):
    if not n_filters > 0:
      raise ValueError(
          'Number of filters must be positive, not {0:%d}'.format(n_filters))
    if max_freq > sample_rate // 2:
      raise ValueError('Upper frequency {0:d} exceeds Nyquist {1:d}'.format(
          int(max_freq), int(sample_rate // 2)))
    self.n_filters = n_filters
    self.min_freq = min_freq
    self.max_freq = max_freq
    self.sample_rate = sample_rate
    self.window_len = window_len
    self.window_stride = window_stride
    self.n_fft = n_fft
    self.normalize_energy = normalize_energy
    self.filt_edge = None
    self.mel_filters = None
    self.center_frequencies = None
    self.fwhms = None
    self._build_mels()
    self._build_gabors()

  def _gabor_wavelet(self, center_frequency: float, fwhm: float):
    """Creates a gabor wavelet centered in frequency center_frequency and with full-width at half maximum fwhm."""
    tmax = self.window_len * self.sample_rate / 1000

    def gabor_function(t):
      return (1 / (np.sqrt(2 * np.pi) * fwhm)) * np.exp(
          1j * center_frequency * t) * np.exp(-t**2 / (2 * fwhm**2))

    return np.asarray(
        [gabor_function(t) for t in np.arange(-tmax / 2, tmax / 2 + 1)])

  def _gabor_params_from_mel(self, mel_filter):
    """Infers, given a mel-filter, the corresponding center frequencyand full-width at half-maximum, in radians."""
    coeff = np.sqrt(2 * np.log(2)) * self.n_fft
    mel_filter = np.sqrt(mel_filter)
    center_frequency = np.argmax(mel_filter)
    peak = mel_filter[center_frequency]
    half_magnitude = peak / 2.0
    spread = np.where(mel_filter >= half_magnitude)[0]
    width = max(spread[-1] - spread[0], 1)
    return center_frequency * 2 * np.pi / self.n_fft, coeff / (np.pi * width)

  def _melfilter_energy(self, mel_filter):
    """Computes the energy of a mel-filter (area under the magnitude spectrum)."""
    height = max(mel_filter)
    hz_spread = (len(np.where(mel_filter > 0)[0]) + 2) * 2 * np.pi / self.n_fft
    return 0.5 * height * hz_spread

  def _build_mels(self):
    """Creates a bank of mel-filters.

    This functions creates a bank of mel-filters characterized by the class
    instance' attributes (n_filters, min_freq, max_freq, etc.)

    """
    # build mel filter matrix
    self.mel_filters = [
        np.zeros(self.n_fft // 2 + 1) for _ in range(self.n_filters)
    ]
    dfreq = self.sample_rate / self.n_fft

    melmax = utils.hz2mel(self.max_freq)
    melmin = utils.hz2mel(self.min_freq)
    dmelbw = (melmax - melmin) / (self.n_filters + 1)
    # filter edges in hz
    filt_edge = utils.mel2hz(melmin +
                             dmelbw * np.arange(self.n_filters + 2, dtype='d'))
    self.filt_edge = filt_edge
    for filter_idx in range(self.n_filters):
      # Filter triangles in dft points
      left_fr = min(round(filt_edge[filter_idx] / dfreq), self.n_fft // 2)
      centerfr = min(round(filt_edge[filter_idx + 1] / dfreq), self.n_fft // 2)
      right_fr = min(round(filt_edge[filter_idx + 2] / dfreq), self.n_fft // 2)
      height = 1
      if centerfr != left_fr:
        left_slope = height / (centerfr - left_fr)
      else:
        left_slope = 0
      freq = left_fr + 1
      while freq < centerfr:
        self.mel_filters[filter_idx][int(freq)] = (freq - left_fr) * left_slope
        freq += 1
      if freq == centerfr:
        self.mel_filters[filter_idx][int(freq)] = height
        freq += 1
      if centerfr != right_fr:
        right_slope = height / (centerfr - right_fr)
      while freq < right_fr:
        self.mel_filters[filter_idx][int(freq)] = (freq -
                                                   right_fr) * right_slope
        freq += 1
      if self.normalize_energy:
        energy = self._melfilter_energy(self.mel_filters[filter_idx])
        self.mel_filters[filter_idx] /= energy

  def _build_gabors(self):
    """Creates a bank of gabor filters.

    This functions first extracts the center frequency and FWHM of
    mel-filterbanks. It then uses these parameters to create matching gabor
    filters.

    """
    self.center_frequencies = []
    self.fwhms = []
    self.gabor_filters = []
    for mel_filter in self.mel_filters:
      center_frequency, fwhm = self._gabor_params_from_mel(mel_filter)
      self.fwhms.append(fwhm)
      self.center_frequencies.append(center_frequency)
      gabor_filter = self._gabor_wavelet(center_frequency, fwhm)
      # Renormalizes the gabor wavelets
      gabor_filter = gabor_filter * np.sqrt(
          self._melfilter_energy(mel_filter) * 2 * np.sqrt(np.pi) * fwhm)
      self.gabor_filters.append(gabor_filter)
