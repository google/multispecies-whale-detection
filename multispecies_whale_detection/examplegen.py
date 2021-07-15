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
import os
from typing import BinaryIO, Dict, Iterable, Optional, Tuple, TypeVar, Union

import apache_beam as beam
from apache_beam.io import fileio
from dataclasses import dataclass
from dateutil import parser as date_parser
from dateutil import tz
import intervaltree
from multispecies_whale_detection import dataset
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
    parsed = parsed.replace(tzinfo=tz.UTC)
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


def _restrict_to_clip(
    begin: datetime.timedelta, end: datetime.timedelta,
    clip_metadata: ClipMetadata
) -> Optional[Tuple[datetime.timedelta, datetime.timedelta]]:
  """Restricts an interval to the duration from ClipMetadata.

  Args:
    begin: The start of the interval, relative to the clip. May be negative.
    end: The end of the interval, relative to the clip. May be negative.
    clip_metadata: Description of the clip.

  Returns:
    Endpoints of the interval restricted to the clip as nonnegative timedeltas
    or None if the interval does not intersect the clip.
  """
  begin = max(begin, datetime.timedelta(seconds=0))
  end = min(end, clip_metadata.duration)
  if begin < clip_metadata.duration and end > datetime.timedelta(seconds=0):
    return (begin, end)
  else:
    return None


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
class FileAnnotation(Annotation):
  """Annotation as time differences from the start of a file."""
  begin: datetime.timedelta
  end: datetime.timedelta

  def relative_endpoints(
      self, clip_metadata: ClipMetadata
  ) -> Optional[Tuple[datetime.timedelta, datetime.timedelta]]:
    """Expresses this annotation as an interval within a given clip.

    Args:
      clip_metadata: Description of the clip, including its position relative to
        the file it came from.

    Returns:
      Begin and end offsets from the start of the clip or None if the annotation
      does not overlap the clip.
    """
    return _restrict_to_clip(
        self.begin - clip_metadata.start_relative_to_file,
        self.end - clip_metadata.start_relative_to_file,
        clip_metadata,
    )


@dataclass
class UTCAnnotation(Annotation):
  """Annotation whose endpoints are absolute datetimes.

  To avoid ambiguity, the datetimes must be time zone aware.

  Development note: This isn't used at this revision. See TODO tagged
  [utc_endpoints] that defers to another (soon) change to keep the size
  of changes manageable.
  """
  begin: datetime.datetime
  end: datetime.datetime

  def __init__(self, label, begin, end):
    if not (begin.tzinfo and end.tzinfo):
      raise ValueError('endpoint datetimes must be time zone aware')
    self.label = label
    self.begin = begin
    self.end = end

  def relative_endpoints(
      self, clip_metadata: ClipMetadata
  ) -> Optional[Tuple[datetime.timedelta, datetime.timedelta]]:
    """Expresses this annotation as an interval within a given clip.

    Args:
      clip_metadata: Description of the clip. start_utc must be set.

    Returns:
      Begin and end offsets from the start of the clip or None if the annotation
      does not overlap the clip.

    Raises:
      ValueError if the clip_metadata has start_utc unset.
    """
    return _restrict_to_clip(
        self.begin - clip_metadata.start_utc,
        self.end - clip_metadata.start_utc,
        clip_metadata,
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
    return datetime.timedelta(
        days=days, seconds=seconds, microseconds=microseconds)

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
    utc = instance.astimezone(tz.UTC)
    return self._tuple_coder.encode((utc.year, utc.month, utc.day, utc.hour,
                                     utc.minute, utc.second, utc.microsecond))

  def decode(self, encoded):
    return datetime.datetime(*self._tuple_coder.decode(encoded), tzinfo=tz.UTC)


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


def audio_example(waveform: np.array, sample_rate: int,
                  annotations: Iterable[Annotation], filename: str,
                  channel: int, offset_seconds: float) -> tf.train.Example:
  """Constructs a TensorFlow Example with labeled audio.

  Args:
    waveform: 'audio_raw_pcm16' bytes feature with this raw, 16-bit,
      little-endian PCM audio.
    sample_rate: 'sample_rate' float feature scalar with the sample rate for
      waveform.
    annotations: 'annotation_begin', 'annotation_end', and 'annotation_label'
      features give endpoints of the annotations on waveform in corresponding
      indices.
    filename: 'filename' bytes feature with the full path to the source audio
      file, for reference.
    channel: 'channel' int64 feature indicates the channel index from the source
      audio.
    offset_seconds: 'offset_seconds' float feature scalar with the offset of the
      start of the chunk for this Example from the start of the audio file. (For
      duty-cycled XWAV files, this will be computed in UTC, not seconds from the
      start of the file.)

  Returns:
    A TensorFlow Example with features as documented in the Args section.
  """
  example = tf.train.Example()
  features = example.features.feature

  features[dataset.Features.AUDIO.value.name].bytes_list.value.append(
      waveform.astype('<i2').tobytes())
  features[dataset.Features.SAMPLE_RATE.value.name].int64_list.value.append(
      sample_rate)
  features[dataset.Features.FILENAME.value.name].bytes_list.value.append(
      filename.encode())
  features[dataset.Features.CHANNEL.value.name].int64_list.value.append(channel)
  features[dataset.Features.OFFSET.value.name].float_list.value.append(
      offset_seconds)

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


def read_annotations(
    readable_file: BinaryIO) -> Iterable[Tuple[str, Annotation]]:
  """Parses an annotations CSV file.

  See py:meth:Annotation.parse_csv_row for a description of the format.

  Args:
    readable_file: Binary file-like object positioned at the beginning of the
      CSV.

  Yields:
    Pairs of filename and parsed Annotation.
  """
  with readable_file.open() as infile:
    reader = csv.DictReader(io.TextIOWrapper(infile))
    for row in reader:
      # TODO(mattharvey): [utc_endpoints] Read UTC annotations when we don't
      # have both 'begin' and 'end' columns and handle arbitrary mixtures of
      # UTC and file annotations in the join on filename (make_audio_examples).
      annotation = FileAnnotation(
          begin=datetime.timedelta(seconds=float(row['begin'])),
          end=datetime.timedelta(seconds=float(row['end'])),
          label=row['label'],
      )
      yield (row['filename'], annotation)


def make_audio_examples(keyed_join_result: Tuple[str, JoinResult],
                        clip_duration_seconds=10.0,
                        resample_rate=16000) -> Iterable[tf.train.Example]:
  """Converts audio/annotation join to TensorFlow Examples.

  This is the core audio processing method of this pipeline. Given a join of
  exactly one audio stream to zero or more annotations, it reads the audio
  stream one clip at a time, expresses the endpoints of the annotations for that
  clip as seconds relative to the clip start, and emits the labeled clip as a
  TensorFlow Example.

  Args:
    keyed_join_result: A pair of a fully-qualified path to an audio file and a
      JoinResult. The JoinResult is a dict with keys 'audio' and 'annotations'.
      The 'audio' key maps to at most one file reader to be handled by
      :py:mod:`soundfile`. The 'annotations' key maps to zero or more Annotation
      objects corresponding to the same fully-qualified path as the audio
      stream.
    clip_duration_seconds: The intended duration of the audio clip in each
      emitted Example.
    resample_rate: Sample rate for the audio in the emitted Examples. The input
      audio stream will be resampled if the sample rate does not match.

  Yields:
    tf.train.Example with annotated PCM audio. For the feature specification of
    these Examples, see :py:func:`audio_example`.
  """
  filename, join_result = keyed_join_result
  del filename  # not needed. We get it from readable_file instead.

  readable_file = _only_element(join_result['audio'])
  if not readable_file:
    beam.metrics.Metrics.counter('examplegen', 'audio_file_not_found').inc()
    return

  annotations_tree = intervaltree.IntervalTree()
  for annotation in join_result['annotations']:
    annotations_tree[annotation.begin:annotation.end] = annotation

  with readable_file.open() as infile:
    reader = soundfile.SoundFile(infile)
    clip_duration_samples = np.round(clip_duration_seconds *
                                     reader.samplerate).astype(int)
    while reader.tell() + clip_duration_samples < reader.frames:
      offset_seconds = reader.tell() / reader.samplerate
      pcm_audio = resampy.resample(
          reader.read(clip_duration_samples, dtype='int16', always_2d=True),
          reader.samplerate,
          resample_rate,
          axis=0,
      )

      annotations_relative_to_clip = []

      clip_begin = datetime.timedelta(seconds=offset_seconds)
      clip_end = clip_begin + datetime.timedelta(seconds=clip_duration_seconds)

      for interval in annotations_tree[clip_begin:clip_end]:
        annotation = interval.data
        annotations_relative_to_clip.append(
            FileAnnotation(
                begin=annotation.begin - clip_begin,
                end=min(annotation.end - clip_begin,
                        datetime.timedelta(seconds=clip_duration_seconds)),
                label=annotation.label))

      for channel in range(reader.channels):
        # TODO(mattharvey): Option for annotations to pertain to either or all
        # channels or a specific channel.
        yield audio_example(
            waveform=pcm_audio[:, channel],
            sample_rate=resample_rate,
            annotations=annotations_relative_to_clip,
            filename=readable_file.metadata.path,
            channel=channel,
            offset_seconds=offset_seconds,
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
      clip_duration_seconds=configuration.clip_duration_seconds,
      resample_rate=configuration.resample_rate,
  )

  with beam.Pipeline(options=options) as pipeline_root:
    all_files = pipeline_root | 'ListFiles' >> fileio.MatchFiles(
        configuration.input_directory + '/**')
    audio_files = all_files | 'MatchAudio' >> extension_filter(
        {'.wav', '.flac'})
    csv_files = all_files | 'MatchCsv' >> extension_filter({'.csv'})

    audio_streams = (
        audio_files | 'ReadAudio' >> fileio.ReadMatches()
        | 'KeyAudioByFilename' >> beam.Map(lambda r: (r.metadata.path, r)))
    annotations = (
        csv_files | 'ReadCsv' >> fileio.ReadMatches()
        | 'ParseCsv' >> beam.ParDo(read_annotations))
    labeled_streams = ({
        'audio': audio_streams,
        'annotations': annotations,
    } | 'JoinOnFilename' >> beam.CoGroupByKey())

    examples = labeled_streams | 'MakeExample' >> beam.FlatMap(
        bind_make_audio_examples)
    # To make sure training examples within a batch are as close as possible to
    # being independent, order them randomly.
    examples = examples | beam.Reshuffle()
    _ = examples | 'WriteRecords' >> beam.io.tfrecordio.WriteToTFRecord(
        os.path.join(configuration.output_directory, 'tfrecords'),
        coder=beam.coders.ProtoCoder(tf.train.Example))

    # TODO(mattharvey): Implement customized text formatting for metadata.csv.
    _ = audio_files | 'WriteListing' >> beam.io.textio.WriteToText(
        os.path.join(configuration.output_directory, 'audio_files'))
