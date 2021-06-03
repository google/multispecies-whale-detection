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

  FILENAME = Feature('source_source_filename',
                     tf.io.FixedLenFeature([], tf.string))
  CHANNEL = Feature('source_channel', tf.io.FixedLenFeature([], tf.int64))
  OFFSET = Feature('source_offset_seconds', tf.io.FixedLenFeature([],
                                                                  tf.float32))
  # TODO(mattharvey): Add 'source_utc' float32 once the reader implements UTC
  # timebase.

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
