# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Command line utility to print audio TFRecord files.

The utility uses the feature specification from this :py:mod:`dataset` module,
including the conversion of the waveform feature 'audio_raw_pcm16' from bytes to
a float list.

This can be useful for debugging when making changes to the parsing logic or
example generator pipeline.
"""

import argparse
import itertools
import sys
from typing import Sequence

from multispecies_whale_detection import dataset
import tensorflow as tf


def run(argv: Sequence[str]) -> None:
  """Runs the audio TFRecord debugging utility."""
  arg_parser = argparse.ArgumentParser(
      description='TFRecord inspection utility')
  arg_parser.add_argument(
      '--tfrecord_filepattern',
      type=str,
      help='Path to or pattern matching the TFRecord file(s) to inspect',
      required=True,
  )
  arg_parser.add_argument(
      '--limit',
      type=int,
      help='Number of records to inspect',
      default=10,
  )
  args = arg_parser.parse_args(argv[1:])

  input_dataset = dataset.new(args.tfrecord_filepattern)
  for features in itertools.islice(input_dataset, args.limit):
    for key, value in features.items():
      if isinstance(value, tf.sparse.SparseTensor):
        value = value.values
      print(key + ':')
      print(value)
    print()


if __name__ == '__main__':
  run(sys.argv)
