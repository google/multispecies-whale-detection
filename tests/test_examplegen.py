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

import datetime
from typing import Optional, Tuple
import unittest

import apache_beam as beam
from dateutil import tz
from multispecies_whale_detection import examplegen


def relative_endpoints_to_file_annotation_fixture(
    begin_seconds: float,
    end_seconds: float) -> Optional[examplegen.ClipAnnotation]:
  """Template testing FileAnnotations relative to a fixture clip.

  The fixture clip covers the time interval [10s, 20s] relative to the file.

  Args:
    begin_seconds: Beginning of the test FileAnnotation relative to the file.
    end_seconds: Ending of the test FileAnnotation relative to the file.

  Returns:
    Annotation over the restriction of the given time interval to the fixture
    clip or None if they don't overlap.
  """
  start_relative_to_file = datetime.timedelta(seconds=20)

  clip_metadata = examplegen.ClipMetadata(
      filename='test.wav',
      sample_rate=10000,
      duration=datetime.timedelta(seconds=10),
      index_in_file=2,
      start_relative_to_file=start_relative_to_file,
      start_utc=None,
  )
  annotation = examplegen.FileAnnotation(
      begin=datetime.timedelta(seconds=begin_seconds),
      end=datetime.timedelta(seconds=end_seconds),
      label='Oo',
  )
  return annotation.make_relative(clip_metadata)


def relative_endpoints_to_utc_annotation_fixture(
    begin_seconds: float,
    end_seconds: float) -> Optional[examplegen.ClipAnnotation]:
  """Template testing UTCAnnotations relative to a fixture clip.

  This is a modification of the FileAnnotation version to convert identical
  offset arguments (relative to the start of the file) to the corresponding UTC
  times, based on a hard-coded fixture file_start_utc.

  The fixture clip covers the time interval [10s, 20s] relative to the file.

  Args:
    begin_seconds: Beginning of the test UTCAnnotation relative to the file.
    end_seconds: Ending of the test UTCAnnotation relative to the file.

  Returns:
    Annotation over the restriction of the given time interval to the fixture
    clip or None if they don't overlap.
  """
  file_start_utc = datetime.datetime(2012, 2, 3, 11, 45, 20, tzinfo=tz.UTC)
  start_relative_to_file = datetime.timedelta(seconds=20)

  clip_metadata = examplegen.ClipMetadata(
      filename='test.wav',
      sample_rate=10000,
      duration=datetime.timedelta(seconds=10),
      index_in_file=2,
      start_relative_to_file=start_relative_to_file,
      start_utc=file_start_utc + start_relative_to_file,
  )
  annotation = examplegen.UTCAnnotation(
      begin=file_start_utc + datetime.timedelta(seconds=begin_seconds),
      end=file_start_utc + datetime.timedelta(seconds=end_seconds),
      label='Oo',
  )
  return annotation.make_relative(clip_metadata)


class TestAnnotations(unittest.TestCase):

  def test_parse_annotation_rel_file(self):
    annotation = examplegen.Annotation.parse_csv_row({
        'label': 'Mn',
        'begin': '1.2',
        'end': '1.8',
    })
    expected = examplegen.FileAnnotation(
        label='Mn',
        begin=datetime.timedelta(seconds=1.2),
        end=datetime.timedelta(seconds=1.8),
    )
    self.assertEqual(expected, annotation)

  def test_parse_annotation_utc(self):
    annotation = examplegen.Annotation.parse_csv_row({
        'label': 'Mn',
        'begin_utc': '2008-05-06 11:24:41.268000',
        'end_utc': '2008-05-06 11:24:42.472000',
    })
    expected = examplegen.UTCAnnotation(
        label='Mn',
        begin=datetime.datetime(2008, 5, 6, 11, 24, 41, 268000, tzinfo=tz.UTC),
        end=datetime.datetime(2008, 5, 6, 11, 24, 42, 472000, tzinfo=tz.UTC),
    )
    self.assertEqual(expected, annotation)

  def test_file_endpoints_override_utc_endpoints(self):
    annotation = examplegen.Annotation.parse_csv_row({
        'label': 'Mn',
        'begin': '1.2',
        'end': '1.8',
        'begin_utc': '2008-05-06 11:24:41.268000',
        'end_utc': '2008-05-06 11:24:42.472000',
    })
    expected = examplegen.FileAnnotation(
        label='Mn',
        begin=datetime.timedelta(seconds=1.2),
        end=datetime.timedelta(seconds=1.8),
    )
    self.assertEqual(expected, annotation)

  def test_parse_annotation_missing_fields(self):
    with self.assertRaises(ValueError):
      examplegen.Annotation.parse_csv_row({'label': 'Mn'})

  def test_round_trip_file_annotation(self):
    annotation = examplegen.FileAnnotation(
        label='Mn',
        begin=datetime.timedelta(seconds=2.1),
        end=datetime.timedelta(seconds=3.1),
    )
    coder = beam.coders.registry.get_coder(examplegen.Annotation)

    encoded = coder.encode(annotation)

    self.assertEqual(annotation, coder.decode(encoded))

  def test_file_annotation_relative_endpoints_within_clip(self):
    clip_annotation = relative_endpoints_to_file_annotation_fixture(22, 23.2)
    self.assertEqual(datetime.timedelta(seconds=2), clip_annotation.begin)
    self.assertEqual(datetime.timedelta(seconds=3.2), clip_annotation.end)

  def test_file_annotation_relative_endpoints_before_clip(self):
    self.assertIsNone(relative_endpoints_to_file_annotation_fixture(1.5, 2.5))

  def test_file_annotation_relative_endpoints_after_clip(self):
    self.assertIsNone(relative_endpoints_to_file_annotation_fixture(42, 45))

  def test_file_annotation_relative_endpoints_overlap_begin(self):
    clip_annotation = relative_endpoints_to_file_annotation_fixture(19.5, 22.1)
    self.assertEqual(datetime.timedelta(seconds=0), clip_annotation.begin)
    self.assertEqual(datetime.timedelta(seconds=2.1), clip_annotation.end)

  def test_round_trip_utc_annotation(self):
    begin = datetime.datetime(2012, 2, 3, 11, 45, 15, tzinfo=tz.UTC)
    end = begin + datetime.timedelta(seconds=1.7)
    annotation = examplegen.UTCAnnotation(label='Mn', begin=begin, end=end)
    coder = beam.coders.registry.get_coder(examplegen.Annotation)

    encoded = coder.encode(annotation)

    self.assertEqual(annotation, coder.decode(encoded))

  def test_utc_annotation_relative_endpoints_within_clip(self):
    clip_annotation = relative_endpoints_to_utc_annotation_fixture(22, 23.2)
    self.assertEqual(datetime.timedelta(seconds=2), clip_annotation.begin)
    self.assertEqual(datetime.timedelta(seconds=3.2), clip_annotation.end)

  def test_utc_annotation_relative_endpoints_before_clip(self):
    self.assertIsNone(relative_endpoints_to_utc_annotation_fixture(1.5, 2.5))

  def test_utc_annotation_relative_endpoints_after_clip(self):
    self.assertIsNone(relative_endpoints_to_utc_annotation_fixture(42, 45))

  def test_utc_annotation_relative_endpoints_overlap_begin(self):
    clip_annotation = relative_endpoints_to_utc_annotation_fixture(19.5, 22.1)
    self.assertEqual(datetime.timedelta(seconds=0), clip_annotation.begin)
    self.assertEqual(datetime.timedelta(seconds=2.1), clip_annotation.end)


if __name__ == '__main__':
  unittest.main()
