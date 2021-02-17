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
"""Utility functions for LEAF."""

import enum
import gin
import numpy as np


def hz2mel(f: float):
  """Hertz to mel scale conversion.

  Converts a frequency from hertz to mel values.

  Args:
    f: a frequency in Hz

  Returns:
    value of f on a mel scale
  """
  return 2595 * np.log10(1 + f / 700)


def mel2hz(m: float):
  """Mel to hertz conversion.

  Converts a frequency from mel to hertz values.

  Args:
    m: a frequency in mels

  Returns:
    value of m on the linear frequency scale
  """
  return 700 * (np.power(10, m / 2595) - 1)


@gin.constants_from_enum
class WindowType(enum.Enum):
  HAMMING = 1
  HANNING = 2
  SQUARED_HANNING = 3


def window(window_type: WindowType, tmax: int):
  """Window functions generator.

  Creates a window of type window_type and duration tmax.
  Currently, hanning (also known as Hann) and hamming windows are available.

  Args:
    window_type: str, type of window function (hanning, squared_hanning,
      hamming)
    tmax: int, duration of the window, in samples

  Returns:
    a window function as np array
  """

  def hanning(n: int):
    return 0.5 * (1 - np.cos(2 * np.pi * (n - 1) / (tmax - 1)))

  def hamming(n: int):
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / tmax)

  if window_type == WindowType.HANNING:
    return np.asarray([hanning(n) for n in range(tmax)])
  elif window_type == WindowType.SQUARED_HANNING:
    return np.asarray([hanning(n) for n in range(tmax)])**2
  elif window_type == WindowType.HAMMING:
    return np.asarray([hamming(n) for n in range(tmax)])
  else:
    raise ValueError('Wrong window type.')
