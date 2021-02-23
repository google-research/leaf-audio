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

# python3
"""Basic test case for running a single training step."""

import os.path

from absl import flags
from absl import logging
from absl.testing import parameterized
import gin
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from example import train

FLAGS = flags.FLAGS
CONFIG_FOLDER = os.path.join(os.path.dirname(__file__), 'configs/')
DEFAULT_CONFIG = 'leaf'


def _parse_gin_config(config=DEFAULT_CONFIG, override=None):
  filename = os.path.join(CONFIG_FOLDER, config + '.gin')
  with open(filename, 'rt') as file_handle:
    gin.parse_config(file_handle)
  gin.parse_config("""
      ConvNet.filters = [64, 128]
  """)
  if override is not None:
    gin.parse_config(override)


class TrainTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    gin.clear_config()

  @parameterized.named_parameters(
      ('leaf', 'leaf'),
      ('leaf_custom', 'leaf_custom'),
      ('mel', 'mel'),
      ('sincnet', 'sincnet'),
      ('sincnet_plus', 'sincnet_plus'),
      ('tfbanks', 'tfbanks'),
  )
  def test_config(self, config):
    _parse_gin_config(config=config)
    self._run_single_epoch()

  def _run_single_epoch(self):
    num_examples = 100

    def as_dataset(self, *args, **kwargs):
      return tf.data.Dataset.from_generator(
          lambda: ({
              'audio': np.ones(shape=(16000,), dtype=np.uint16),
              'label': i % 12,
          } for i in range(num_examples)),
          output_types=self.info.features.dtype,
          output_shapes=self.info.features.shape,
      )

    model_dir = self.create_tempdir().full_path
    with tfds.testing.mock_data(
        num_examples=num_examples, as_dataset_fn=as_dataset):
      train.train(
          workdir=model_dir,
          dataset='speech_commands',
          num_epochs=1,
      )

    files_in_model_dir = tf.io.gfile.listdir(model_dir)
    logging.info('files_in_model_dir: %s', files_in_model_dir)
    some_expected_files = ['checkpoint.index', 'checkpoint.data-00000-of-00001']
    self.assertAllInSet(some_expected_files, files_in_model_dir)


if __name__ == '__main__':
  tf.test.main()
