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
"""For FFT-based 1D convolutions."""

from absl.testing import parameterized
from leaf_audio import convolution
import tensorflow.compat.v2 as tf


class ConvTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    tf.random.set_seed(0)

  @parameterized.parameters([[1000, 32, 1, 4], [1000, 256, 4, 8]])
  def test_fft_conv1d(self, seq_len, filter_len, batch_size, num_filters):
    inputs = tf.sort(tf.random.normal(shape=(batch_size, seq_len, 1)), axis=1)
    filters = tf.random.normal(shape=(filter_len, 1, num_filters))

    target = tf.nn.convolution(inputs, filters, padding='SAME')
    outputs, filters_l1 = convolution.fft_conv1d(inputs, filters)

    self.assertEqual(outputs.shape, target.shape)
    self.assertEqual(outputs.shape, (batch_size, seq_len, num_filters))
    self.assertEqual(filters_l1.shape, ())

    k = filter_len // 2
    self.assertAllClose(outputs[:, k:-k, :], target[:, k:-k, :], atol=0.01)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
