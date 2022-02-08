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

"""The main script to run the training loop."""

from typing import Sequence

from absl import app
from absl import flags
import gin
from example import train


flags.DEFINE_multi_string(
    'gin_config', [], 'List of paths to the config files.')
flags.DEFINE_multi_string(
    'gin_bindings', [], 'Newline separated list of Gin parameter bindings.')
FLAGS = flags.FLAGS


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_bindings)
  train.train()


if __name__ == '__main__':
  app.run(main)
