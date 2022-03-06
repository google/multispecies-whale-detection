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

import dataclasses
import datetime
import math
import tempfile
from typing import Iterable
import unittest

import numpy as np
import tensorflow as tf

from multispecies_whale_detection import dataset
from multispecies_whale_detection import examplegen

CLASS_NAMES = ['Bm', 'Bp', 'Be', 'Mn', 'Eg', 'Ej', 'Oo']
"""Whale genus / species codes."""


def _window_count(clip_duration: float, window_duration: float,
                  hop: float) -> int:
  """Returns the number context window hops in bounds of a clip."""
  return int((clip_duration - window_duration) / hop) + 1


def _sin_waveform(frequency_hz: float, duration: float,
                  sample_rate: int) -> tf.Tensor:
  """Returns a sinusoidal waveform with the given shape parameter arguments."""
  sample_index = tf.range(0, duration * sample_rate, dtype=tf.float32)
  return tf.math.sin(2 * math.pi * frequency_hz / sample_rate * sample_index)


def _waveform_contains_window(window: tf.Tensor, waveform: tf.Tensor) -> bool:
  """Tests if an extracted window matches any slice of a given waveform."""
  # Slide a window of the same duration as the extracted window over the
  # original waveform and check that there exists a sample offset at which it
  # matches the extracted window.
  clip_duration_samples = waveform.shape[0]
  window_duration_samples = window.shape[0]
  num_positions = clip_duration_samples - window_duration_samples
  all_windows = tf.stack(
      [waveform[i:i + window_duration_samples] for i in range(num_positions)])
  tiled_extracted_window = tf.tile(
      tf.expand_dims(window, 0), [num_positions, 1])
  extracted_matches_row = tf.math.reduce_all(
      all_windows == tiled_extracted_window, axis=1)
  return tf.math.reduce_any(extracted_matches_row).numpy()


def _temp_tfrecords(
    examples: Iterable[tf.train.Example]) -> tempfile.NamedTemporaryFile:
  """Returns a temporary TFRecord file, which serializes the given examples."""
  # pylint: disable=consider-using-with
  # This is a factory for the caller to use with.
  file = tempfile.NamedTemporaryFile()
  with tf.io.TFRecordWriter(file.name) as writer:
    for example in examples:
      writer.write(example.SerializeToString())
  return file


@dataclasses.dataclass(frozen=True)
class Example:
  """Structured arguments to examplegen.audio_example.

  This is intended to enable more readable assertions by providing fine-grained
  access to a fixture example populated with default values.
  """
  sample_rate: int = 200
  duration: float = 10.0

  clip_metadata: examplegen.ClipMetadata = examplegen.ClipMetadata(
      filename='audio.wav',
      sample_rate=sample_rate,
      duration=datetime.timedelta(seconds=duration),
      index_in_file=123,
      start_relative_to_file=datetime.timedelta(seconds=0.0),
      start_utc=datetime.datetime(2022, 3, 4, 22, 36, 39),
  )

  waveform: np.ndarray = (_sin_waveform(
      frequency_hz=440.0,
      duration=duration,
      sample_rate=sample_rate,
  ).numpy() * (np.iinfo(np.int16).max - 1)).astype(np.int16)

  channel: int = 0

  annotations: Iterable[examplegen.ClipAnnotation] = dataclasses.field(
      default_factory=lambda: [
          examplegen.ClipAnnotation(
              begin=datetime.timedelta(seconds=3.62),
              end=datetime.timedelta(seconds=4.1),
              label='Eg')
      ])


def _integration_test_fixture_examples() -> Iterable[tf.train.Example]:
  """Returns contents for a TFRecord file read by tests that require one."""
  fixture = Example()
  yield examplegen.audio_example(
      clip_metadata=fixture.clip_metadata,
      waveform=fixture.waveform,
      sample_rate=fixture.sample_rate,
      channel=fixture.channel,
      annotations=fixture.annotations,
  )


class TestDataset(unittest.TestCase):

  def assert_features_match_fixture(self, features):
    fixture = Example()

    np.testing.assert_allclose(
        fixture.waveform, (np.iinfo(np.int16).max - 1) *
        features[dataset.Features.AUDIO.value.name],
        atol=2.0)
    self.assertEqual(fixture.sample_rate,
                     features[dataset.Features.SAMPLE_RATE.value.name])
    self.assertEqual(fixture.channel,
                     features[dataset.Features.CHANNEL.value.name])

    self.assertEqual(fixture.clip_metadata.filename,
                     features[dataset.Features.FILENAME.value.name])
    self.assertEqual(
        fixture.clip_metadata.start_relative_to_file.total_seconds(),
        features[dataset.Features.START_RELATIVE_TO_FILE.value.name])
    self.assertEqual(fixture.clip_metadata.start_utc.timestamp(),
                     features[dataset.Features.START_UTC.value.name])

    annotation = fixture.annotations[0]
    self.assertEqual(
        [annotation.begin.total_seconds()],
        features[dataset.Features.ANNOTATION_BEGIN.value.name].values)
    self.assertEqual(
        [annotation.end.total_seconds()],
        features[dataset.Features.ANNOTATION_END.value.name].values)
    self.assertEqual(
        [annotation.label],
        features[dataset.Features.ANNOTATION_LABEL.value.name].values)

  def test_parse_fn(self):
    self.assert_features_match_fixture(
        dataset.parse_fn(
            next(iter(
                _integration_test_fixture_examples())).SerializeToString()))

  def test_new(self):
    examples = _integration_test_fixture_examples()
    with _temp_tfrecords(examples) as infile:
      new_dataset = dataset.new(infile.name)
      features = next(iter(new_dataset))
    self.assert_features_match_fixture(features)

  def test_extract_window_waveform(self):
    sample_rate = 2000
    waveform = _sin_waveform(
        frequency_hz=440.0,
        duration=2.0,
        sample_rate=sample_rate,
    )
    features = {
        dataset.Features.AUDIO.value.name: waveform,
        dataset.Features.SAMPLE_RATE.value.name: sample_rate,
    }
    # Note that this doubles a test for the case where ANNOTATION features are
    # not provided.
    start = 0.5
    window_duration = 0.2
    begin_sample = int(start * sample_rate)
    end_sample = begin_sample + int(window_duration * sample_rate)
    expected_extract = waveform[begin_sample:end_sample]

    extract, _ = dataset.extract_window(
        features,
        start=start,
        duration=window_duration,
        class_names=CLASS_NAMES,
    )

    self.assertTrue(tf.reduce_all(expected_extract == extract))

  def test_extract_window_labels(self):
    features = {
        dataset.Features.ANNOTATION_BEGIN.value.name:
            tf.sparse.from_dense([0.5, 2.5, 3.5]),
        dataset.Features.ANNOTATION_END.value.name:
            tf.sparse.from_dense([1.1, 3.2, 4.0]),
        dataset.Features.ANNOTATION_LABEL.value.name:
            tf.sparse.from_dense(['Oo', 'Mn', 'Ej']),
    }
    # Note that this doubles a test for the case where AUDIO features are
    # not provided.
    _, labels = dataset.extract_window(
        features,
        start=0.0,
        duration=3.0,
        class_names=CLASS_NAMES,
        min_overlap=0.1,
    )

    self.assertEqual([len(CLASS_NAMES)], labels.shape)
    self.assertEqual(['Mn', 'Oo'], list(tf.boolean_mask(CLASS_NAMES, labels)))

  def test_extract_window_label_overlap_too_short(self):
    features = {
        dataset.Features.ANNOTATION_BEGIN.value.name:
            tf.sparse.from_dense([0.0]),
        dataset.Features.ANNOTATION_END.value.name:
            tf.sparse.from_dense([1.05]),
        dataset.Features.ANNOTATION_LABEL.value.name:
            tf.sparse.from_dense(['Bm']),
    }

    _, labels = dataset.extract_window(
        features,
        start=1.0,
        duration=1.0,
        class_names=CLASS_NAMES,
        min_overlap=0.1,
    )

    self.assertEqual([len(CLASS_NAMES)], labels.shape)
    self.assertTrue(tf.math.reduce_all(labels == 0.0))

  def test_single_random_window(self):
    sample_rate = 200
    window_duration = 1.0
    # This will extract a single window from a clip slightly longer than the
    # extracted window, so that a label interval in the middle is guaranteed to
    # overlap.
    clip_duration = window_duration + 0.1
    waveform = _sin_waveform(
        frequency_hz=440.0,
        duration=clip_duration,
        sample_rate=sample_rate,
    )
    sample_size = 1
    class_names = ['Bm', 'Eg']
    features = {
        dataset.Features.AUDIO.value.name:
            waveform,
        dataset.Features.SAMPLE_RATE.value.name:
            sample_rate,
        dataset.Features.ANNOTATION_BEGIN.value.name:
            tf.sparse.from_dense([0.5]),
        dataset.Features.ANNOTATION_END.value.name:
            tf.sparse.from_dense([0.6]),
        dataset.Features.ANNOTATION_LABEL.value.name:
            tf.sparse.from_dense(['Bm']),
    }

    windowing = dataset.RandomWindowing(sample_size)
    windows, labels = windowing.extract_windows(
        features,
        window_duration,
        class_names=class_names,
    )

    window_duration_samples = int(sample_rate * window_duration)
    self.assertEqual(
        tf.TensorShape([sample_size, window_duration_samples]), windows.shape)
    self.assertEqual(
        tf.TensorShape([sample_size, len(class_names)]), labels.shape)
    self.assertTrue(tf.math.reduce_all([[1.0, 0.0]] == labels))
    self.assertTrue(_waveform_contains_window(windows[0], waveform))

  def test_random_windows_count(self):
    sample_size = 3
    sample_rate = 200
    window_duration = 1.0
    waveform = _sin_waveform(
        frequency_hz=440.0,
        duration=2.0,
        sample_rate=sample_rate,
    )
    features = {
        dataset.Features.AUDIO.value.name: waveform,
        dataset.Features.SAMPLE_RATE.value.name: sample_rate,
    }
    class_names = ['Oo']

    windowing = dataset.RandomWindowing(sample_size)
    windows, labels = windowing.extract_windows(
        features,
        duration=window_duration,
        class_names=class_names,
    )

    window_duration_samples = int(window_duration * sample_rate)
    self.assertEqual(
        tf.TensorShape([sample_size, window_duration_samples]), windows.shape)
    self.assertEqual(
        tf.TensorShape([sample_size, len(class_names)]), labels.shape)

  def test_sliding_windows(self):
    sample_rate = 200
    clip_duration = 10.0
    window_duration = 2.0
    hop = window_duration / 2
    waveform = _sin_waveform(
        frequency_hz=440.0,
        duration=clip_duration,
        sample_rate=sample_rate,
    )
    window_duration_samples = int(sample_rate * window_duration)
    # Label that intersects windows[1] and windows[2].
    class_names = ['Oo']
    features = {
        dataset.Features.AUDIO.value.name:
            waveform,
        dataset.Features.SAMPLE_RATE.value.name:
            sample_rate,
        dataset.Features.ANNOTATION_BEGIN.value.name:
            tf.sparse.from_dense([2.25]),
        dataset.Features.ANNOTATION_END.value.name:
            tf.sparse.from_dense([2.75]),
        dataset.Features.ANNOTATION_LABEL.value.name:
            tf.sparse.from_dense(['Oo']),
    }

    windowing = dataset.SlidingWindowing(hop)
    windows, labels = windowing.extract_windows(features, window_duration,
                                                class_names)

    expected_window_count = _window_count(clip_duration, window_duration, hop)
    self.assertEqual(
        tf.TensorShape([expected_window_count, window_duration_samples]),
        windows.shape)
    self.assertEqual(
        tf.TensorShape([expected_window_count,
                        len(class_names)]), labels.shape)

  def test_new_window_dataset_random(self):
    window_duration = 2.0
    sample_size = 3
    class_names = ['Orca', 'Offshore', 'Resident']
    examples = _integration_test_fixture_examples()
    with _temp_tfrecords(examples) as infile:
      for batch_size in [2, 4]:

        window_dataset = dataset.new_window_dataset(
            tfrecord_filepattern=infile.name,
            duration=window_duration,
            windowing=dataset.RandomWindowing(sample_size),
            class_names=class_names,
        ).batch(batch_size)

        windows, labels = next(iter(window_dataset))
        window_duration_samples = int(window_duration * Example().sample_rate)
        num_outputs = min(batch_size, sample_size)
        self.assertEqual(
            tf.TensorShape([num_outputs, window_duration_samples]),
            windows.shape)
        self.assertEqual(
            tf.TensorShape([num_outputs, len(class_names)]), labels.shape)

  def test_new_window_dataset_sliding(self):
    window_duration = 1.0
    hop = 0.5
    class_names = ['Orca', 'Offshore', 'Resident']
    examples = _integration_test_fixture_examples()
    with _temp_tfrecords(examples) as infile:

      for batch_size in [2, 128]:
        window_dataset = dataset.new_window_dataset(
            tfrecord_filepattern=infile.name,
            duration=window_duration,
            windowing=dataset.SlidingWindowing(hop),
            class_names=class_names,
        ).batch(batch_size)

        windows, labels = next(iter(window_dataset))
        example = Example()
        clip_duration = len(example.waveform) / example.sample_rate
        num_hops = _window_count(clip_duration, window_duration, hop)
        window_duration_samples = int(window_duration * Example().sample_rate)
        num_outputs = min(batch_size, num_hops)
        self.assertEqual(
            tf.TensorShape([num_outputs, window_duration_samples]),
            windows.shape)
        self.assertEqual(
            tf.TensorShape([num_outputs, len(class_names)]), labels.shape)


if __name__ == '__main__':
  unittest.main()
