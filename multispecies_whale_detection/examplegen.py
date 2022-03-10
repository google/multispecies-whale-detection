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
"""Example Generator Pipeline.

The run method implements a Beam pipeline that generates TFRecord files for
training and evaluation by joining audio files, using the full filename, to
annotations from CSV files, reading the audio files into short clips, and
serializing the labeled clips as TensorFlow Examples.
"""

import csv
import datetime
import functools
import io
import itertools
import os
from typing import BinaryIO, Dict, Iterable, Optional, Tuple, TypeVar, Union

import apache_beam as beam
from apache_beam.io import fileio
from dataclasses import dataclass
from dateutil import parser as date_parser
import intervaltree
from multispecies_whale_detection import dataset
from multispecies_whale_detection import xwav
import numpy as np
import resampy
import soundfile
import tensorflow as tf

T = TypeVar('T')


def _only_element(iterable: Iterable[T], default_value: T = None) -> T:
  """Unwraps an iterator that is expected to have at most one element.

  Args:
    iterable: An Iterable with at most one element.
    default_value: The value to return when the Iterable has no elements.

  Returns:
    The unique element of the Iterable or the default value when the Iterable is
    empty.

  Raises:
    ValueError: if the Iterable has more than one element.
  """
  iterator = iter(iterable)
  element = next(iterator, default_value)
  try:
    _ = next(iterator)
    raise ValueError('Iterable had more than one element')
  except StopIteration:
    pass
  return element


def _parse_utc(text: str) -> datetime.datetime:
  """Leniently parses a datetime, defaulting to UTC if no zone is provided."""
  parsed = date_parser.parse(text)
  if not parsed.tzinfo:
    parsed = parsed.replace(tzinfo=datetime.timezone.utc)
  return parsed


@dataclass
class Configuration:
  """Application settings for the pipeline."""
  input_directory: str
  output_directory: str
  clip_duration_seconds: float = 10
  resample_rate: int = 16000


@dataclass
class ClipMetadata:
  """Description of a clip of audio from a longer file."""
  filename: str
  sample_rate: int
  duration: datetime.timedelta
  index_in_file: int
  start_relative_to_file: datetime.timedelta
  start_utc: Optional[datetime.datetime]


# to annotate a factory method
AnnotationType = TypeVar('AnnotationType', bound='Annotation')


@dataclass
class Annotation:
  """Value object for a labeled time interval within some audio.

  Depending on context, begin and end may be interpreted as relative to the
  start of an audio file, a clip taken from a longer file, or the UNIX epoch.
  """
  label: str

  @classmethod
  def parse_csv_row(cls, row: Dict[str, str]) -> AnnotationType:
    """Parses an annotation from a CSV row.

    Args:
      row: Dictionary mapping CSV headers to field values. The headers must
        include "label" and one of ("begin" / "end" floating-point endpoints in
        seconds relative to the start of the file) or ("begin_utc" / "end_utc"
        endpoints as absolute times as strings in any format that
        dateutil.parser will convert to an aware datetime). When both are
        present, only the endpoints relative to the start of the file are used.

    Returns:
      Annotation parsed from the CSV row.

    Raises:
      ValueError if the given row does not include all the required fields.
    """
    label = row.get('label', None)
    if not label:
      raise ValueError('label was empty or not provided')
    begin_rel_file = row.get('begin', None)
    end_rel_file = row.get('end', None)
    if begin_rel_file and end_rel_file:
      return FileAnnotation(
          label=row['label'],
          begin=datetime.timedelta(seconds=float(begin_rel_file)),
          end=datetime.timedelta(seconds=float(end_rel_file)),
      )
    begin_text = row.get('begin_utc', None)
    end_text = row.get('end_utc', None)
    if begin_text and end_text:
      return UTCAnnotation(
          label=row['label'],
          begin=_parse_utc(begin_text),
          end=_parse_utc(end_text),
      )
    raise ValueError('row should have either both "begin" and "end" fields '
                     'or both "begin_utc" and "end_utc" fields')


@dataclass
class ClipAnnotation(Annotation):
  """Annotation as time differences from the start of a clip.

  This type is used for annotations relative to an extracted clip of audio
  intended to be included in its entirety into a TensorFlow Example.

  This is in contrast to FileAnnotation (below). There, begin / end are in a
  timeline whose zero point is at the start of the file. Here, the zero point
  is at the beginning of a clip which often has been extracted from a longer
  file.

  audio_example packs these ClipAnnotations into parallel "annotation" fields
  of a TensorFlow Example.
  """
  begin: datetime.timedelta
  end: datetime.timedelta


def _restrict_to_clip(
    begin: datetime.timedelta,
    end: datetime.timedelta,
    clip_metadata: ClipMetadata,
    label: str,
) -> Optional[ClipAnnotation]:
  """Restricts an interval to the duration from ClipMetadata.

  Args:
    begin: The start of the interval, relative to the clip. May be negative.
    end: The end of the interval, relative to the clip. May be negative.
    clip_metadata: Description of the clip.
    label: Label to set in the returned ClipAnnotation.

  Returns:
    ClipAnnotation with the intersection of (begin, end) with the clip described
    by clip_metadata or None if that interstion is empty.
  """
  assert end > begin
  begin = max(begin, datetime.timedelta(seconds=0))
  end = min(end, clip_metadata.duration)
  if begin < clip_metadata.duration and end > datetime.timedelta(seconds=0):
    return ClipAnnotation(begin=begin, end=end, label=label)
  else:
    return None


@dataclass
class FileAnnotation(Annotation):
  """Annotation as time differences from the start of a file."""
  begin: datetime.timedelta
  end: datetime.timedelta

  def make_relative(self,
                    clip_metadata: ClipMetadata) -> Optional[ClipAnnotation]:
    """Expresses this annotation as an interval within a given clip.

    Args:
      clip_metadata: Description of the clip, including its position relative to
        the file it came from.

    Returns:
      An annotation relative to the given clip or None if there is no overlap.
    """
    return _restrict_to_clip(
        self.begin - clip_metadata.start_relative_to_file,
        self.end - clip_metadata.start_relative_to_file,
        clip_metadata,
        self.label,
    )


@dataclass
class UTCAnnotation(Annotation):
  """Annotation whose endpoints are absolute datetimes.

  To avoid ambiguity, the datetimes must be time zone aware.
  """
  begin: datetime.datetime
  end: datetime.datetime

  def __init__(self, label, begin, end):
    if not (begin.tzinfo and end.tzinfo):
      raise ValueError('endpoint datetimes must be time zone aware')
    self.label = label
    self.begin = begin
    self.end = end

  def make_relative(self,
                    clip_metadata: ClipMetadata) -> Optional[ClipAnnotation]:
    """Expresses this annotation as an interval within a given clip.

    Args:
      clip_metadata: Description of the clip. start_utc must be set.

    Returns:
      An annotation relative to the given clip or None if there is no overlap.

    Raises:
      ValueError if clip_metadata.start_utc is None.
    """
    return _restrict_to_clip(
        self.begin - clip_metadata.start_utc,
        self.end - clip_metadata.start_utc,
        clip_metadata,
        self.label,
    )


# Type of the values for the CoGroupByKey (filename) done by this pipeline.
# See later make_audio_examples, which processes this JoinResult.
JoinResult = Dict[str, Union[Iterable[fileio.ReadableFile],
                             Iterable[Annotation]]]


class TimedeltaCoder(beam.coders.Coder):
  """Compact Beam Coder for datetime.timedelta."""

  def __init__(self):
    int_coder = beam.coders.VarIntCoder()
    self._tuple_coder = beam.coders.TupleCoder(
        (int_coder, int_coder, int_coder))

  def encode(self, instance):
    return self._tuple_coder.encode(
        (instance.days, instance.seconds, instance.microseconds))

  def decode(self, encoded):
    days, seconds, microseconds = self._tuple_coder.decode(encoded)
    return datetime.timedelta(days=days,
                              seconds=seconds,
                              microseconds=microseconds)

  def is_deterministic(self):
    return True


class UTCDatetimeCoder(beam.coders.Coder):
  """Beam Coder that codes aware datetimes as UTC tuples."""

  def __init__(self):
    int_coder = beam.coders.VarIntCoder()
    self._tuple_coder = beam.coders.TupleCoder(
        (int_coder, int_coder, int_coder, int_coder, int_coder, int_coder,
         int_coder))

  def encode(self, instance):
    utc = instance.astimezone(datetime.timezone.utc)
    return self._tuple_coder.encode((utc.year, utc.month, utc.day, utc.hour,
                                     utc.minute, utc.second, utc.microsecond))

  def decode(self, encoded):
    return datetime.datetime(*self._tuple_coder.decode(encoded),
                             tzinfo=datetime.timezone.utc)


class AnnotationCoder(beam.coders.Coder):
  """Compact Beam Coder for Annotations."""

  FILE_TYPE_CODE = 1
  UTC_TYPE_CODE = 2

  def __init__(self):
    int_coder = beam.coders.VarIntCoder()  # type code
    bytes_coder = beam.coders.BytesCoder()
    self._base_coder = beam.coders.TupleCoder((int_coder, bytes_coder))

    timedelta_coder = TimedeltaCoder()
    self._file_coder = beam.coders.TupleCoder(
        (timedelta_coder, timedelta_coder, beam.coders.StrUtf8Coder()))

    datetime_coder = UTCDatetimeCoder()
    self._utc_coder = beam.coders.TupleCoder(
        (datetime_coder, datetime_coder, beam.coders.StrUtf8Coder()))

  def encode(self, annotation):
    if isinstance(annotation, FileAnnotation):
      type_code = self.FILE_TYPE_CODE
      sub_coder = self._file_coder
    elif isinstance(annotation, UTCAnnotation):
      type_code = self.UTC_TYPE_CODE
      sub_coder = self._utc_coder
    else:
      raise TypeError('unknown annotation type')
    sub_encoded = sub_coder.encode(
        (annotation.begin, annotation.end, annotation.label))
    return self._base_coder.encode((type_code, sub_encoded))

  def decode(self, encoded):
    type_code, sub_encoded = self._base_coder.decode(encoded)
    if type_code == self.FILE_TYPE_CODE:
      begin, end, label = self._file_coder.decode(sub_encoded)
      return FileAnnotation(begin=begin, end=end, label=label)
    elif type_code == self.UTC_TYPE_CODE:
      begin, end, label = self._utc_coder.decode(sub_encoded)
      return UTCAnnotation(begin=begin, end=end, label=label)

  def is_deterministic(self):
    return True


beam.coders.registry.register_coder(Annotation, AnnotationCoder)


def read_annotations(infile: BinaryIO) -> Iterable[Tuple[str, Annotation]]:
  """Parses an annotations CSV file.

  See py:meth:Annotation.parse_csv_row for a description of the format.

  Args:
    infile: Binary file-like object positioned at the beginning of the CSV.

  Yields:
    Pairs of filename and parsed Annotation.
  """
  reader = csv.DictReader(io.TextIOWrapper(infile))
  for row in reader:
    yield (row['filename'], Annotation.parse_csv_row(row))


def beam_read_annotations(readable_file: fileio.ReadableFile):
  """Opens the file and calls read_annotations."""
  return read_annotations(readable_file.open())


def generate_clips(
    filename: str, infile: BinaryIO, clip_duration: datetime.timedelta
) -> Iterable[Tuple[ClipMetadata, np.array]]:
  """Reads a file and generates equal-length clips and metadata.

  In general the file may be much longer than the requested clip duration. The
  start of the clip advances by the clip duration (disjoint tiling) until the
  file (or subchunk, in the XWAV case) is exhausted.

  This allows both XWAV and non-XWAV files to be treated as the same type by
  calling code, despite the fact that XWAVs are in effect a collection of
  shorter (~75s) files.

  Args:
    filename: Passed through to ClipMetadata.
    infile: Seekable file-like object in any audio format supported by
      soundfile. Optional XWAV headers will be used to populate
      ClipMetadata.start_utc.
    readable_file: File handle of the type supplied by Beam.
    clip_duration: The desired length of each clip. When this does not evenly
      divide the duration of a subchunk (or whole file in the non-XWAV case),
      the remaining audio will be discarded.

  Yields:
    Pairs of clip metadata and NumPy arrays of audio of shape
      (samples, channels).
  """
  try:
    infile.seek(0)
    xwav_reader = xwav.Reader(infile)
    sample_rate = xwav_reader.header.fmt_chunk.sample_rate
    clip_duration_samples = np.round(clip_duration.total_seconds() *
                                     sample_rate).astype(int)
    # TODO(mattharvey): Add the ability to specify hop size as a field of
    # examplegen.Configuration and pass that field, or perhaps the whole
    # Configuration, through to here.
    hop = clip_duration_samples

    # For ClipMetadata.start_relative_to_file, because clips may not exactly
    # fill the subchunk, we need to increment the subchunk start relative to
    # the file in an outer loop over subchunks.
    subchunk_rel_file = datetime.timedelta(seconds=0)

    clip_index_in_file = 0
    for subchunk, samples in xwav_reader:
      subchunk_duration_samples = samples.shape[0]
      for begin, end in zip(
          range(0, subchunk_duration_samples, hop),
          range(clip_duration_samples, subchunk_duration_samples, hop)):
        clip_rel_subchunk = datetime.timedelta(seconds=begin / sample_rate)
        clip_metadata = ClipMetadata(
            filename=filename,
            sample_rate=sample_rate,
            duration=clip_duration,
            index_in_file=clip_index_in_file,
            start_relative_to_file=(subchunk_rel_file + clip_rel_subchunk),
            start_utc=(subchunk.time + clip_rel_subchunk),
        )
        clip_samples = samples[begin:end, :]
        yield (clip_metadata, clip_samples)
        clip_index_in_file += 1
      subchunk_rel_file += datetime.timedelta(
          seconds=subchunk_duration_samples / sample_rate)
  except xwav.Error:
    # TODO(matharvey): Consider refactoring this by adding an abstraction layer
    # around both SoundFile and xwav.Reader, which would present a single API
    # here and give an extension point for new reader implementations. Do not
    # forget that the non-XWAV branch may need to work with hours-long files
    # that are too big to pre-read into memory, nor that the non-XWAV branch
    # does not implement hop size yet.
    infile.seek(0)
    reader = soundfile.SoundFile(infile)
    sample_rate = reader.samplerate
    clip_duration_samples = np.round(clip_duration.total_seconds() *
                                     sample_rate).astype(int)

    # SoundFile read defaults to continuing where it left off, implying that the
    # hop is always exactly the duration of the context window in this, the
    # non-XWAV case.
    clip_index_in_file = 0
    while reader.tell() + clip_duration_samples < reader.frames:
      clip_rel_file = datetime.timedelta(seconds=reader.tell() /
                                         reader.samplerate)
      clip_metadata = ClipMetadata(
          filename=filename,
          sample_rate=sample_rate,
          duration=clip_duration,
          index_in_file=clip_index_in_file,
          start_relative_to_file=clip_rel_file,
          start_utc=None,
      )
      clip_samples = reader.read(clip_duration_samples,
                                 dtype='int16',
                                 always_2d=True)
      yield (clip_metadata, clip_samples)
      clip_index_in_file += 1


def audio_example(clip_metadata: ClipMetadata, waveform: np.array,
                  sample_rate: int, channel: int,
                  annotations: Iterable[ClipAnnotation]) -> tf.train.Example:
  """Constructs a TensorFlow Example with labeled audio.

  Args:
    clip_metadata: Passed through to multiple features: 'filename' bytes feature
      with the full path to the source audio file, for reference;
      'start_relative_to_file' float feature scalar with the offset of waveform
      from the start of the file; 'start_utc' float feature with seconds since
      the UNIX epoch until the start of waveform, missing when the original data
      does not provide timestamps.
    waveform: 'audio_raw_pcm16' bytes feature with this raw, 16-bit,
      little-endian PCM audio.
    sample_rate: 'sample_rate' float feature scalar with the sample rate for
      waveform. When the waveform has been resampled, this will not match
      clip_metadata.sample_rate, which pertains to the original file.
    channel: 'channel' int64 feature indicates the channel index from the source
      audio.
    annotations: 'annotation_begin', 'annotation_end', and 'annotation_label'
      features are parallel arrays, with each entry corresponding to one of
      these given annotations.

  Returns:
    A TensorFlow Example with features as documented in the Args section.
  """
  example = tf.train.Example()
  features = example.features.feature

  features[dataset.Features.AUDIO.value.name].bytes_list.value.append(
      waveform.astype('<i2').tobytes())
  features[dataset.Features.SAMPLE_RATE.value.name].int64_list.value.append(
      sample_rate)
  features[dataset.Features.CHANNEL.value.name].int64_list.value.append(channel)

  features[dataset.Features.FILENAME.value.name].bytes_list.value.append(
      clip_metadata.filename.encode())
  features[dataset.Features.START_RELATIVE_TO_FILE.value.
           name].float_list.value.append(
               clip_metadata.start_relative_to_file.total_seconds())
  if clip_metadata.start_utc:
    features[dataset.Features.START_UTC.value.name].float_list.value.append(
        clip_metadata.start_utc.timestamp())

  for annotation in annotations:
    features[
        dataset.Features.ANNOTATION_BEGIN.value.name].float_list.value.append(
            annotation.begin.total_seconds())
    features[
        dataset.Features.ANNOTATION_END.value.name].float_list.value.append(
            annotation.end.total_seconds())
    features[
        dataset.Features.ANNOTATION_LABEL.value.name].bytes_list.value.append(
            annotation.label.encode())

  return example


class AnnotationTrees:

  def __init__(self, annotations: Iterable[Annotation]):
    self._file_tree = intervaltree.IntervalTree()
    self._utc_tree = intervaltree.IntervalTree()
    self._empty_count = 0
    for annotation in annotations:
      if annotation.end <= annotation.begin:
        self._empty_count += 1
        continue
      is_utc = isinstance(annotation, UTCAnnotation)
      is_file = isinstance(annotation, FileAnnotation)
      if is_utc:
        self._utc_tree[annotation.begin:annotation.end] = annotation
      elif is_file:
        self._file_tree[annotation.begin:annotation.end] = annotation
      else:
        assert is_utc or is_file

  def annotate_clip(self,
                    clip_metadata: ClipMetadata) -> Iterable[ClipAnnotation]:
    file_intervals = self._file_tree[clip_metadata.start_relative_to_file:(
        clip_metadata.start_relative_to_file + clip_metadata.duration)]
    if clip_metadata.start_utc:
      utc_intervals = self._utc_tree[clip_metadata.start_utc:(
          clip_metadata.start_utc + clip_metadata.duration)]
    else:
      utc_intervals = []
    for interval in itertools.chain(iter(file_intervals), iter(utc_intervals)):
      annotation = interval.data
      clip_annotation = annotation.make_relative(clip_metadata)
      if clip_annotation:
        yield clip_annotation


def make_audio_examples(
    keyed_join_result: Tuple[str, JoinResult],
    clip_duration: datetime.timedelta,
    resample_rate: int = 16000) -> Iterable[tf.train.Example]:
  """Converts audio/annotation join to TensorFlow Examples.

  This is the core method of this pipeline. Given a join of exactly one audio
  stream to zero or more annotations, it reads the audio stream one clip at a
  time, expresses the endpoints of the annotations for that clip as seconds
  relative to the clip start, and emits the labeled clip as a TensorFlow
  Example.

  Args:
    keyed_join_result: A pair of a fully-qualified path to an audio file and a
      JoinResult. The JoinResult is a dict with keys 'audio' and 'annotations'.
      The 'audio' key maps to at most one file reader to be handled by
      :py:mod:`soundfile`. The 'annotations' key maps to zero or more Annotation
      objects corresponding to the same fully-qualified path as the audio
      stream.
    clip_duration: The intended duration of the audio clip in each emitted
      Example.
    resample_rate: Sample rate for the audio in the emitted Examples. The input
      audio stream will be resampled if the sample rate does not match.

  Yields:
    tf.train.Example with annotated PCM audio. For the feature specification of
    these Examples, see :py:func:`audio_example`.
  """
  filename, join_result = keyed_join_result
  del filename  # Trust readable_file more.

  readable_file = _only_element(join_result['audio'])
  if not readable_file:
    beam.metrics.Metrics.counter('examplegen', 'audio_file_not_found').inc()
    return
  filename = readable_file.metadata.path

  annotation_trees = AnnotationTrees(join_result['annotations'])

  with readable_file.open() as infile:
    for clip_metadata, clip_samples in generate_clips(filename, infile,
                                                      clip_duration):
      if np.round(clip_metadata.sample_rate) == np.round(resample_rate):
        pcm_audio = clip_samples
      else:
        pcm_audio = resampy.resample(
            clip_samples,
            clip_metadata.sample_rate,
            resample_rate,
            axis=0,
        )

      clip_annotations = annotation_trees.annotate_clip(clip_metadata)

      for channel, waveform in enumerate(pcm_audio.T):
        # TODO(mattharvey): Option for annotations to pertain to either or all
        # channels or a specific channel.
        beam.metrics.Metrics.counter('examplegen', 'examples-generated').inc()
        yield audio_example(
            clip_metadata=clip_metadata,
            waveform=waveform,
            sample_rate=resample_rate,
            channel=channel,
            annotations=clip_annotations,
        )


def extension_filter(kept_extensions: Iterable[str]) -> beam.PTransform:
  """Returns a Beam filter that keeps strs with given file extensions."""

  def keep_fn(file_metadata: beam.io.filesystem.FileMetadata) -> bool:
    _, extension = os.path.splitext(file_metadata.path)
    return extension in kept_extensions

  return beam.Filter(keep_fn)


def run(configuration: Configuration,
        options: beam.options.pipeline_options.PipelineOptions) -> None:
  """Runs the examplegen Beam pipeline.

  Args:
    configuration: Input and output paths and settings related to feature
      extraction.
    options: Settings related to the Beam runner. (See beam.apache.org.)

  Returns:
    None
  """
  bind_make_audio_examples = functools.partial(
      make_audio_examples,
      clip_duration=datetime.timedelta(
          seconds=configuration.clip_duration_seconds),
      resample_rate=configuration.resample_rate,
  )

  with beam.Pipeline(options=options) as pipeline:
    all_files = pipeline | 'ListFiles' >> fileio.MatchFiles(
        configuration.input_directory + '/**')
    audio_files = all_files | 'MatchAudio' >> extension_filter(
        {'.wav', '.flac'})
    csv_files = all_files | 'MatchCsv' >> extension_filter({'.csv'})

    audio_streams = (
        audio_files | 'ReadAudio' >> fileio.ReadMatches() |
        'KeyAudioByFilename' >> beam.Map(lambda r: (r.metadata.path, r)))
    annotations = (csv_files | 'ReadCsv' >> fileio.ReadMatches() |
                   'ParseCsv' >> beam.ParDo(beam_read_annotations))
    labeled_streams = ({
        'audio': audio_streams,
        'annotations': annotations,
    } | 'JoinOnFilename' >> beam.CoGroupByKey())

    examples = labeled_streams | 'MakeExample' >> beam.FlatMap(
        bind_make_audio_examples)
    # To make sure training examples within a batch are as close as possible to
    # being independent, shuffle at the level of the entire pipeline run.
    examples = examples | beam.Reshuffle()
    _ = examples | 'WriteRecords' >> beam.io.tfrecordio.WriteToTFRecord(
        os.path.join(configuration.output_directory, 'tfrecords'),
        coder=beam.coders.ProtoCoder(tf.train.Example))

    # TODO(mattharvey): Implement customized text formatting for metadata.csv.
    _ = audio_files | 'WriteListing' >> beam.io.textio.WriteToText(
        os.path.join(configuration.output_directory, 'audio_files'))

    return pipeline.run()
