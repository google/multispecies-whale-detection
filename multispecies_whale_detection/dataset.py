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
import abc
import dataclasses
import enum
import functools
from typing import Dict, Sequence, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

DEFAULT_MIN_OVERLAP = 0.1


@dataclasses.dataclass
class Feature:
  """Pairing of the name and parsing specification for a feature."""
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


FeaturesType = Dict[str, Union[tf.Tensor, tf.sparse.SparseTensor]]


def _audio_duration(features: FeaturesType) -> tf.Tensor:
  """Returns the duration, in seconds, of the audio in a Features dict."""
  waveform = features[Features.AUDIO.value.name]
  sample_rate = tf.cast(features[Features.SAMPLE_RATE.value.name], tf.float32)
  return tf.cast(tf.shape(waveform)[0], tf.float32) / sample_rate


def parse_fn(serialized_example: bytes) -> FeaturesType:
  """Parses and converts Tensors for this module's Features.

  This casts the audio_raw_pcm16 feature to float32 and scales it into the range
  [-1.0, 1.0].

  Args:
    serialized_example: A serialized tf.train.ExampleProto with the features
      dict keys declared in the :py:class:Features enum.

  Returns:
    Tensor-valued dict of features. The keys are those declared in the
    :py:class:Features enum.
  """
  features = tf.io.parse_single_example(
      serialized_example, {f.value.name: f.value.spec for f in Features})
  audio_key: str = Features.AUDIO.value.name
  features[audio_key] = tf.cast(
      tf.io.decode_raw(features[audio_key], tf.int16), tf.float32) / np.iinfo(
          np.int16).max
  return features


def new(tfrecord_filepattern: str):
  """Creates a Dataset yielding dicts with the schema declared in Features."""
  dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(tfrecord_filepattern))
  dataset = dataset.map(parse_fn)
  return dataset


def _extract_waveform_window(
    features: FeaturesType,
    start: float,
    duration: float,
) -> Optional[tf.Tensor]:
  """Waveform subroutine for extract_window."""
  try:
    waveform = features[Features.AUDIO.value.name]
  except KeyError:
    return None
  sample_rate = tf.cast(features[Features.SAMPLE_RATE.value.name], tf.float32)
  start_sample = tf.cast(tf.math.floor(sample_rate * start), tf.int32)
  # Note that a using a fixed duration rather than an unconstrained [begin,
  # end] avoids flakiness due to float rounding errors.
  duration_samples = tf.cast(tf.math.floor(sample_rate * duration), tf.int32)
  tf.debugging.assert_less_equal(start_sample + duration_samples,
                                 tf.shape(waveform)[0])
  return waveform[start_sample:start_sample + duration_samples]


def _extract_labels_window(features: FeaturesType, start: float,
                           duration: float, class_names: Sequence[str],
                           min_overlap: float) -> tf.Tensor:
  """Labels subroutine for extract_window."""
  try:
    annotation_begin = tf.sparse.to_dense(
        features[Features.ANNOTATION_BEGIN.value.name])
    annotation_end = tf.sparse.to_dense(
        features[Features.ANNOTATION_END.value.name])
    annotation_label = tf.sparse.to_dense(
        features[Features.ANNOTATION_LABEL.value.name])
  except KeyError:
    return tf.zeros([len(class_names)])

  # Creates a Boolean indicator for whether [begin, end] overlaps each
  # annotation interval on the longer segment.
  end = start + duration
  overlap_indicator = ((start <= annotation_end - min_overlap) &
                       (end >= annotation_begin + min_overlap))
  # Creates a matrix where rows correspond to annotation intervals overlapping
  # this window. Each row has values all equal to the label of the
  # annotation for that row.
  classes_present = tf.reshape(
      tf.boolean_mask(annotation_label, overlap_indicator), [-1, 1])
  num_classes_present = tf.shape(classes_present)[0]
  # Special case to avoid returning an empty labels Tensor when the comparison
  # below would be of empty tensors of class labels.
  if num_classes_present == 0:
    return tf.zeros([len(class_names)])
  tiled_classes_present = tf.tile(classes_present, [1, len(class_names)])
  # Creates a same-shaped matrix, where each column is identically equal to
  # the value in class_names at that columns index.
  tiled_classes_to_compare = tf.tile([class_names], [num_classes_present, 1])
  # Compares the above two matrices to get multi-label binary indicators, in
  # rows attributable to the corresponding annotations.
  indicators = (tiled_classes_present == tiled_classes_to_compare)
  # Aggregates such that the window is labeled positive for any
  # classes that were labels of any overlapping annotation interval.
  return tf.cast(tf.reduce_any(indicators, axis=0), tf.float32)


def extract_window(
    features: FeaturesType,
    start: float,
    duration: float,
    class_names: Sequence[str],
    min_overlap: float = DEFAULT_MIN_OVERLAP
) -> Tuple[Optional[tf.Tensor], tf.Tensor]:
  """Extracts a window waveform and labels from an example.

  This operates on single examples, not batches.

  The labels are converted from the sparse form in features to a Tensor of shape
  [num_classes], whose values are indicators of whether an annotation of the
  corresponding class overlaps the interval [start, start + duration].

  A features dict missing both AUDIO and SAMPLE_RATE is supported for just
  converting labels to binary indicators pertaining to a given interval. If
  AUDIO is in the features dict, then SAMPLE_RATE must be also.

  A features dict missing the annotation-related keys is supported for just
  extracting audio from an unlabeled example, but it is recommended to always
  provide the ANNOTATION features, even if the values are empty. If any of the
  ANNOTATION features is provided, all are required.

  Args:
    features: A dict as returned by :py:meth:`parse_fn`. It conforms to the
      schema :py:class:`Features` represents.
    start: Beginning of the window to extract, in seconds from the start of the
      audio in features.
    duration: Duration of the window to extract, in seconds from the start of
      the audio in features.
    class_names: Tensor-like of shape [num_classes] (the same shape as the
      binary class indicators this function returns).
    min_overlap: Lower bound for duration of overlap between an annotation
      interval and an extracted window for the window to be considered positive
      for the class in that annotation's label.

  Returns:
    Waveform and labels Tensors, where the waveform is a slice of the audio from
    features and where the labels is a [len(class_names)]-dimensional tensor of
    float32 indicators of classes on annotations overlapping the time interval
    of the waveform.
  """
  return (_extract_waveform_window(features, start, duration),
          _extract_labels_window(features, start, duration, class_names,
                                 min_overlap))


class Windowing(abc.ABC):
  """Strategy for slicing fixed-duration windows from an audio clip."""

  @abc.abstractmethod
  def extract_windows(
      self,
      features: FeaturesType,
      duration: float,
      class_names: Sequence[str],
      min_overlap=DEFAULT_MIN_OVERLAP,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Extracts labeled windows.

    Args:
      features: Features of an audio clip as returned by :py:meth:parse_fn.
      duration: Desired duration of extracted windows. All audio tensors
        returned will have exactly this duration.
      windowing: Criteria for choosing start times and number of the extracted
        windows.
      class_names: A list of all possible values of ANNOTATION_LABEL, also known
        as the label vocabulary.
      min_overlap: Lower bound for duration of overlap between an annotation
        interval and an extracted window for the window to be considered
        positive for the class in that annotation's label.

    Returns:
      Audio tensor of shape [sample_size, audio_duration_samples] and labels
      tensor of float32 class labels of shape [num_windows, len(class_names)].
    """
    return NotImplemented


class RandomWindowing(Windowing):
  """Windowing strategy that samples in-bounds windows with random starts."""

  def __init__(self, count: int) -> None:
    self.count = count

  def extract_windows(
      self,
      features: FeaturesType,
      duration: float,
      class_names: Sequence[str],
      min_overlap=DEFAULT_MIN_OVERLAP,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Implements Windowing.extract_window."""
    extracted_windows = []
    extracted_labels = []
    begin_limit = _audio_duration(features) - duration
    for _ in range(self.count):
      begin = tf.random.uniform([], tf.constant(0.0, tf.float32), begin_limit)
      window, label = extract_window(features, begin, duration, class_names,
                                     min_overlap)
      extracted_windows.append(window)
      extracted_labels.append(label)
    return tf.stack(extracted_windows), tf.stack(extracted_labels)


class SlidingWindowing(Windowing):
  """Windowing strategy that slides the window at a fixed hop."""

  def __init__(self, hop: float) -> None:
    self.hop = hop

  def extract_windows(
      self,
      features: FeaturesType,
      duration: float,
      class_names: Sequence[str],
      min_overlap=DEFAULT_MIN_OVERLAP,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Implements Windowing.extract_window."""
    num_hops = 1 + tf.cast(
        tf.math.floor(
            (_audio_duration(features) - duration) / self.hop), tf.int32)
    extracted_windows = tf.TensorArray(tf.float32, size=num_hops)
    extracted_labels = tf.TensorArray(tf.float32, size=num_hops)
    begin = 0.0
    index = 0
    while begin + duration <= _audio_duration(features):
      window, label = extract_window(features, begin, duration, class_names,
                                     min_overlap)
      extracted_windows = extracted_windows.write(index, window)
      extracted_labels = extracted_labels.write(index, label)
      begin += self.hop
      index += 1
    return extracted_windows.stack(), extracted_labels.stack()


def new_window_dataset(
    tfrecord_filepattern: str,
    duration: float,
    class_names: Sequence[str],
    windowing: Windowing,
    min_overlap=DEFAULT_MIN_OVERLAP,
) -> tf.data.Dataset:
  """Parallel tf.data pipeline to extract windows from TFRecords.

  This is intended for initializing a training set when sample_size is not None
  and a validation or test set when hop is not None. Exactly one of the
  two must be None.

  Args:
    tfrecord_filepattern: Glob pattern matching TFRecord files of serialized
      tf.train.Examples as produced by examplegen. These represent mono audio
      clips with sparse labels given by corresponding float endpoints and string
      labels. batch_size
    duration: The common duration of the waveforms this Dataset yields.
    class_names: String values expected in ANNOTATION_LABEL features, the order
      of which determined class indices in the labels this dataset yields.
    sample_size: When set, specifies that this number of windows with random
      starts should be extracted from each input audio. Mutually exclusive with
      hop.
    hop: When set, specifies that all windows sliding over the input audio at
      this period - that remain in bounds - should be extracted. Mutually
      exclusive with sample_size.
    min_overlap: Lower bound on the duration of overlap between an ANNOTATION
      feature interval and a window for that window to count as positive for the
      ANNOTATION_LABEL.

  Returns:
    Dataset of individual (window waveform, label) tuples. All
    windows have the same fixed length, determined by the duration argument.
    The labels are 0.0/1.0 indicators of whether an annotated interval for the
    corresponding class overlaps the window.

  Raises:
    ValueError: If both or neither of sample_size and hop are set.
  """
  assert isinstance(windowing, Windowing)
  window_fn = functools.partial(
      windowing.extract_windows,
      duration=duration,
      class_names=class_names,
      min_overlap=min_overlap,
  )

  def shard_fn(filename):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_fn)
    dataset = dataset.map(window_fn)
    dataset = dataset.unbatch()
    return dataset

  filenames = tf.io.gfile.glob(tfrecord_filepattern)
  dataset = tf.data.Dataset.from_tensor_slices(filenames)
  dataset = dataset.interleave(
      shard_fn,
      cycle_length=len(filenames),
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=False,
  )
  return dataset
