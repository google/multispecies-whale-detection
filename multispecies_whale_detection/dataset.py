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

"""tf.data readers and support for audio TFRecord datasets.

In the :py:class:`Features` enumeration, this module formalizes the specific
feature specification we use within the more loosely-typed
:py:class:`tensorflow.train.Example`.

It also has helper methods for using the tf.data API to read Examples conforming
to this specification, like those produced by
:py:meth:`multispecies_whale_detection.examplegen.pipeline.run`.
"""
import enum
from typing import Dict, Union

from dataclasses import dataclass
import numpy as np
import tensorflow as tf


@dataclass
class Feature:
  name: str
  spec: Union[tf.io.FixedLenFeature, tf.io.VarLenFeature]


class Features(enum.Enum):
  """Feature keys and corresponding parsing configurations for audio clips."""
  AUDIO = Feature('audio_raw_pcm16', tf.io.FixedLenFeature([], tf.string))
  SAMPLE_RATE = Feature('sample_rate', tf.io.FixedLenFeature([], tf.int64))
  CHANNEL = Feature('channel', tf.io.FixedLenFeature([], tf.int64))

  FILENAME = Feature('filename', tf.io.FixedLenFeature([], tf.string))
  START_RELATIVE_TO_FILE = Feature('start_relative_to_file',
                                   tf.io.FixedLenFeature([], tf.float32))
  START_UTC = Feature('start_utc',
                      tf.io.FixedLenFeature([], tf.float32, default_value=-1))

  ANNOTATION_BEGIN = Feature('annotation_begin',
                             tf.io.VarLenFeature(tf.float32))
  ANNOTATION_END = Feature('annotation_end', tf.io.VarLenFeature(tf.float32))
  ANNOTATION_LABEL = Feature('annotation_label', tf.io.VarLenFeature(tf.string))


def parse_fn(
    serialized_example: bytes
) -> Dict[str, Union[tf.Tensor, tf.sparse.SparseTensor]]:
  features = tf.io.parse_single_example(
      serialized_example, {f.value.name: f.value.spec for f in Features})
  audio_key: str = Features.AUDIO.value.name
  features[audio_key] = tf.cast(
      tf.io.decode_raw(features[audio_key], tf.int16), tf.float32) / np.iinfo(
          np.int16).max
  return features


def new(tfrecord_filepattern: str):
  dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(tfrecord_filepattern))
  dataset = dataset.map(parse_fn)
  return dataset
