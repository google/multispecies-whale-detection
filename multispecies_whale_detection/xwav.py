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
"""Support for the XWAV file format.

Many underwater acoustic monitoring datasets are collected on the HARP
(high-frequency acoustic recording package platform).

doi: 10.1109/UT.2007.370760.

Post-processed data from these devices is typically archived in a format called
XWAV.

An XWAV file is a WAV file, but it contains an additional "harp" subchunk that
carries metadata about the deployment as well as a list of "subchunks" that
associate byte locations in the file to UTC start times and durations. This is
useful because HARP recordings typically happen on a duty cycle, for example 5
minutes on / 15 minutes off.

XWAV files have sizes extending into the gigabytes, but they can be thought of
as collections of much shorter WAV files. One common subchunk duration is 75
seconds.

Documentation on the format can be found in Appendix A2 of the Triton User
Manual at

https://github.com/MarineBioAcousticsRC/Triton/blob/master/Extras/TritonUserManual.pdf

MATLAB code for writing headers in this format can be found at

https://github.com/MarineBioAcousticsRC/Triton/blob/master/wrxwavhd.m
"""
import chunk
import dataclasses
import datetime
import enum
import io
import struct
from typing import BinaryIO, Iterable, Sequence, Tuple, TypeVar

import mutagen
from mutagen import flac
import numpy as np
import soundfile


class Error(Exception):
  """Base class exceptions related to the XWAV format."""
  pass


class MissingChunkError(Error):
  """Raised when one of the "fmt " or "harp" chunks is missing."""


class CorruptHeadersError(Error):
  """Raised when XWAV headers are not in the expected format."""
  pass


class OutOfRangeError(Error):
  """Raised when XWAV headers reference an inaccessible file position."""
  pass


class AudioFormat(enum.Enum):
  """Descriptor for how a WAVE file encodes audio.

  See http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
  """
  PCM = 1
  FLOAT = 3
  ALAW = 6
  MULAW = 7
  EXTENSIBLE = 0xfffe


@dataclasses.dataclass(frozen=True)
class FmtChunk:
  """Metadata from the standard WAV "fmt " chunk."""
  audio_format: AudioFormat
  num_channels: int
  sample_rate: int
  bytes_per_second: int
  block_align: int
  bits_per_sample: int

  @classmethod
  def read(cls, reader: BinaryIO):
    """Parses a WAV "fmt " chunk.

    Args:
      reader: file-like object that reads the contents of a "fmt " chunk. (It
        should be positioned at the start of the content, that is after the
        size, of the "fmt " chunk.)

    Returns:
      FmtChunk parsed from the reader.
    """
    (format_code, num_channels, sample_rate, bytes_per_second, block_align,
     bits_per_sample) = struct.unpack_from('<HHLLHH', reader.read(16))
    return FmtChunk(
        audio_format=AudioFormat(format_code),
        num_channels=num_channels,
        sample_rate=sample_rate,
        bytes_per_second=bytes_per_second,
        block_align=block_align,
        bits_per_sample=bits_per_sample,
    )

  def write(self, writer: BinaryIO) -> None:
    """Serializes this in the WAVE format.

    Args:
      writer: file-like object to write to.
    """
    writer.write(
        struct.pack(
            '<HHLLHH',
            self.audio_format.value,
            self.num_channels,
            self.sample_rate,
            self.bytes_per_second,
            self.block_align,
            self.bits_per_sample,
        ))


# For type annotations of factory methods.
SubchunkType = TypeVar('SubchunkType', bound='Subchunk')
HarpChunkType = TypeVar('HarpChunkType', bound='HarpChunk')


@dataclasses.dataclass(frozen=True)
class Subchunk:
  """HARP subchunk metadata, including length, position, and UTC start time."""
  time: datetime
  byte_loc: int
  byte_length: int
  write_length: int
  sample_rate: int
  gain: int

  MICROSECONDS_PER_TICK = 1000  # HARP "ticks" time scale
  ZERO_YEAR = 2000  # offset for XWAV year being unsigned char

  @classmethod
  def read(cls, reader: BinaryIO) -> SubchunkType:
    """Parses an XWAV header with its associated subchunks.

    Args:
      reader: file-like object that reads the contents of a "harp" subchunk. It
        should be positioned at the start of the particular serialized subchunk
        data structure to be read.

    Returns:
      Subchunk dataclass parsed from the reader.
    """
    try:
      year, month, day, hour, minute, second, ticks = struct.unpack(
          '<BBBBBBH', reader.read(8))
      byte_loc, byte_length, write_length, sample_rate = struct.unpack(
          '<IIII', reader.read(16))
      gain = struct.unpack('B7x', reader.read(8))[0]
    except struct.error as e:
      raise CorruptHeadersError from e

    try:
      time = datetime.datetime(cls.ZERO_YEAR + year,
                               month,
                               day,
                               hour,
                               minute,
                               second,
                               ticks * cls.MICROSECONDS_PER_TICK,
                               tzinfo=datetime.timezone.utc)
    except ValueError as e:
      raise CorruptHeadersError from e

    return Subchunk(
        time=time,
        byte_loc=byte_loc,
        byte_length=byte_length,
        write_length=write_length,
        sample_rate=sample_rate,
        gain=gain,
    )

  def write(self, writer: BinaryIO) -> None:
    """Serializes this subchunk in the format used by XWAV headers.

    Args:
      writer: file-like object to write to.
    """
    time = self.time
    writer.write(
        struct.pack(
            '<BBBBBBHIIIIB7x',
            time.year - self.ZERO_YEAR,
            time.month,
            time.day,
            time.hour,
            time.minute,
            time.second,
            time.microsecond // self.MICROSECONDS_PER_TICK,
            self.byte_loc,
            self.byte_length,
            self.write_length,
            self.sample_rate,
            self.gain,
        ))


@dataclasses.dataclass(frozen=True)
class HarpChunk:
  """Metadata from the XWAV-specific "harp" chunk."""
  wav_version_number: int
  firmware_version_number: str
  instrument_id: str
  site_name: str
  experiment_name: str
  disk_sequence_number: int
  disk_serial_number: str
  longitude: float
  latitude: float
  depth: int

  subchunks: Sequence[Subchunk]

  # Decimal degrees per unit of int latitude and longitude
  GEO_SCALE = 100000

  @classmethod
  def serialized_len(cls, num_subchunks: int) -> int:
    """Returns the byte length of a serialized HarpChunk.

    This is a convenience method that saves the caller from needing to know the
    top-level a subchunk record sizes. This can be useful because Subchunk
    byte_loc depends on the value this returns.

    Args:
      num_subchunks: Number of subchunks in the contemplated HarpChunk.
    Retruns: Byte length of the serialization of the contemplated HarpChunk.
    """
    harp_record_bytes = 56
    subchunk_record_bytes = 32
    return harp_record_bytes + subchunk_record_bytes * num_subchunks

  @classmethod
  def read(cls, reader: BinaryIO) -> HarpChunkType:
    """Parses an XWAV header with its associated subchunks.

    Args:
      reader: file-like object that reads the contents of a "harp" chunk. (In
        the context of the WAV (RIFF) file format, it should be positioned at
        the start of the content, that is after the size, of the HARP chunk.)

    Returns:
      HarpChunk dataclass parsed from the reader.
    """
    try:
      wav_version_number = struct.unpack('B', reader.read(1))[0]
      firmware_version_number = _clean_header_str(reader.read(10))
      instrument_id = _clean_header_str(reader.read(4))
      site_name = _clean_header_str(reader.read(4))
      experiment_name = _clean_header_str(reader.read(8))
      disk_sequence_number = struct.unpack('B', reader.read(1))[0]
      disk_serial_number = _clean_header_str(reader.read(8))
      num_raw_files, longitude, latitude, depth = struct.unpack(
          '<HiiH', reader.read(12))
      _ = reader.read(8)  # reserved bytes
    except (struct.error, UnicodeDecodeError) as e:
      raise CorruptHeadersError from e

    latitude = latitude / cls.GEO_SCALE
    longitude = longitude / cls.GEO_SCALE

    subchunks = [Subchunk.read(reader) for _ in range(num_raw_files)]

    return HarpChunk(
        wav_version_number=wav_version_number,
        firmware_version_number=firmware_version_number,
        instrument_id=instrument_id,
        site_name=site_name,
        experiment_name=experiment_name,
        disk_sequence_number=disk_sequence_number,
        disk_serial_number=disk_serial_number,
        longitude=longitude,
        latitude=latitude,
        depth=depth,
        subchunks=subchunks,
    )

  def write(self, writer: BinaryIO) -> None:
    """Serializes this HarpChunk in the format used by XWAV headers.

    Args:
      writer: file-like object to write to.
    """
    writer.write(
        struct.pack(
            '<B10s4s4s8sB8sHiiH8x',
            self.wav_version_number,
            self.firmware_version_number.encode(),
            self.instrument_id.encode(),
            self.site_name.encode(),
            self.experiment_name.encode(),
            self.disk_sequence_number,
            self.disk_serial_number.encode(),
            len(self.subchunks),
            int(self.longitude * self.GEO_SCALE),
            int(self.latitude * self.GEO_SCALE),
            self.depth,
        ))
    for subchunk in self.subchunks:
      subchunk.write(writer)


@dataclasses.dataclass(frozen=True)
class Header:
  """Full parsed XWAV header, including standard WAV headers."""
  fmt_chunk: FmtChunk
  harp_chunk: HarpChunk


def _clean_header_str(value: bytes) -> str:
  """Null-terminates, strips, and removes trailing underscores."""
  return value.split(b'\x00')[0].decode().strip().rstrip('_')


def header_from_chunks(chunks: Iterable[Tuple[str, BinaryIO]]) -> Header:
  """Template method for XWAV header parsing.

  Uncompressed and FLAC-compressed XWAVs need different implementations for
  iterating the WAVE chunks. This method parses the header from a general
  iterable of chunks.

  Args:
    chunks: Iteratable of chunk (id, data) pairs. The data element should be
      positioned at the beginning of the chunk contents as opposed to at the
      chunk id.

  Returns:
    Header dataclass parsed from WAV headers or None if any chunks are missing.

  Raises:
    MissingChunkError: if the WAV headers are valid but the "harp" or "fmt "
      chunk is missing.
  """
  fmt_chunk = None
  harp_chunk = None
  for chunk_id, chunk_bytes in chunks:
    if chunk_id == b'fmt ':
      fmt_chunk = FmtChunk.read(chunk_bytes)
    elif chunk_id == b'harp':
      harp_chunk = HarpChunk.read(chunk_bytes)
    if fmt_chunk and harp_chunk:
      break
  if not (fmt_chunk and harp_chunk):
    raise MissingChunkError
  return Header(fmt_chunk=fmt_chunk, harp_chunk=harp_chunk)


def header_from_wav(reader: BinaryIO) -> Header:
  """Parses an XWAV header from a WAV file.

  Args:
    reader: file-like object positioned at the beginning of an uncompressed XWAV
      file.

  Returns:
    Header dataclass parsed from WAV headers or None if any chunks are missing.

  Raises:
    CorruptHeadersError: if the reader does not read valid WAV file contents.
    MissingChunkError: if the WAV headers are valid but the "harp" or "fmt "
      chunk is missing.
  """
  # Validates that we have a WAV file.
  try:
    riff, riff_size, riff_format = struct.unpack('<4sI4s', reader.read(12))
  except struct.error as e:
    raise CorruptHeadersError from e
  if riff != b'RIFF':
    raise CorruptHeadersError('not in RIFF format')
  del riff_size
  if riff_format != b'WAVE':
    raise CorruptHeadersError('RIFF identifier was not WAVE')

  def generate_chunks() -> Iterable[Tuple[str, BinaryIO]]:
    current_chunk = chunk.Chunk(reader, bigendian=False)
    while current_chunk.getname() != 'data':
      yield current_chunk.getname(), current_chunk
      try:
        current_chunk = chunk.Chunk(reader, bigendian=False)
      except EOFError:
        break

  return header_from_chunks(generate_chunks())


def header_from_flac(reader: BinaryIO) -> Header:
  """Parses an XWAV header from a FLAC file.

  To save storage and bandwidth, XWAV files are sometimes compressed as FLAC
  with the original WAV headers as "foreign metadata," as in the NCEI archive
  (https://console.cloud.google.com/storage/browser/noaa-passive-bioacoustic/pifsc).

  This method reads a HARP Header from the "riff" APPLICATION metadata block.

  Args:
    reader: file-like object positioned at the beginning of a FLAC file whose
      "riff" metadata block contains a "harp" chunk.

  Returns:
    Header dataclass parsed from the APPLICATION metadata or None if that block
    or WAV "fmt " or "harp" chunks are missing.

  Raises:
    CorruptHeadersError: if the expected FLAC headers are missing of invalid.
    MissingChunkError: if an expected chunk is missing from the "riff"
      APPLICATION metadata.
  """
  application_metadata_code = 2

  try:
    flac_reader = flac.FLAC(reader)
  except mutagen.MutagenError as e:
    raise CorruptHeadersError from e

  def generate_chunks() -> Iterable[Tuple[str, BinaryIO]]:
    for metadata_block in flac_reader.metadata_blocks:
      if metadata_block.code != application_metadata_code:
        continue
      try:
        application_metadata_id, chunk_id, size = struct.unpack(
            '<4s4sI', metadata_block.data[:12])
      except struct.error as e:
        raise CorruptHeadersError from e
      if application_metadata_id != b'riff':
        continue
      del size
      chunk_reader = io.BytesIO(metadata_block.data[12:])
      yield chunk_id, chunk_reader

  return header_from_chunks(generate_chunks())


def write(num_channels: int, sample_rate: int, harp_chunk: HarpChunk,
          subchunks: Iterable[Tuple[datetime.datetime, np.ndarray]],
          subchunks_len: int, output: BinaryIO) -> None:
  """Writes an XWAV file.

  The output audio format is 16-bit integer PCM.

  Args:
    num_channels: Number of channels. Set in the output FmtChunk and checked
      against each element in subchunks.
    sample_rate: Audio sample rate. Set in the FmtChunk and each Subchunk in the
      output HarpChunk. Must be the same for each element in subchunks.
    harp_chunk: HarpChunk from which the top-level metadata will be taken. Its
      subchunks property will be ignored in favor of computing the output
      HarpChunk.subchunks from the subchunks Iterable.
    subchunks: Iterable that yields UTC start times and audio with dimensions
      (num_samples, num_channels) starting at those times. num_channels will be
      checked against the one given as an argument and must be the same for the
      entire Iterable. num_samples may be different for each element. The NumPy
      array with the audio must already be of type np.int16 and will not be
      rescaled.
    subchunks_len: The number of elements of the subchunks Iterable. Knowing
      this in advance is required to compute the correct size of the HarpChunk.
    output: Seekable, binary, file-like object into which the output will be
      written.

  Raises:
    ValueError: If the number of elements from iterating subchunks does not
      equal subchunks_len or if any element of subchunks has second dimension
      unequal to num_channels. In either case, bytes might have already been
      written to output at the time of the exception, and the caller should
      delete any destination file.
  """
  bytes_per_sample = 2

  # Reserve bytes for header chunks that will be written last, because some of
  # them depend on or aggregate the sizes of the subchunks to be iterated.
  iff_header_len = 8
  iff_format = b'WAVE'
  riff_len = iff_header_len + len(iff_format)
  fmt_len = 24
  harp_len = iff_header_len + HarpChunk.serialized_len(subchunks_len)
  data_header_len = iff_header_len
  output.write(b'\0' * (riff_len + fmt_len + harp_len + data_header_len))
  data_begin_pos = output.tell()

  harp_chunk_subchunks = []
  for time, samples in subchunks:
    if samples.shape[1] != num_channels:
      raise ValueError('num_channels != samples.shape[1]')
    byte_loc = output.tell()
    if samples.dtype != np.int16:
      raise ValueError('subchunk audio was not np.int16')
    output.write(samples.tobytes())
    byte_length = output.tell() - byte_loc
    # magic copy/paste from wrxwavhd.m (see module docstring)
    write_length = byte_length // (512 - 12)  # sector - 12 bytes of timing
    subchunk = Subchunk(
        time=time,
        byte_loc=byte_loc,
        byte_length=byte_length,
        write_length=write_length,
        sample_rate=sample_rate,
        gain=1,
    )
    harp_chunk_subchunks.append(subchunk)

  if len(harp_chunk_subchunks) != subchunks_len:
    raise ValueError('len(subchunks) (Iterable) != subchunks_len')

  file_size = output.tell()
  output.seek(0)
  output.write(
      struct.pack('<4sI4s', b'RIFF', file_size - iff_header_len, iff_format))
  output.write(struct.pack('<4sI', b'fmt ', fmt_len - iff_header_len))
  FmtChunk(
      audio_format=AudioFormat.PCM,
      num_channels=num_channels,
      sample_rate=sample_rate,
      bytes_per_second=(num_channels * sample_rate * bytes_per_sample),
      block_align=(num_channels * bytes_per_sample),
      bits_per_sample=(bytes_per_sample * 8),
  ).write(output)
  output.write(struct.pack('<4sI', b'harp', harp_len - iff_header_len))
  HarpChunk(
      wav_version_number=harp_chunk.wav_version_number,
      firmware_version_number=harp_chunk.firmware_version_number,
      instrument_id=harp_chunk.instrument_id,
      site_name=harp_chunk.site_name,
      experiment_name=harp_chunk.experiment_name,
      disk_sequence_number=harp_chunk.disk_sequence_number,
      disk_serial_number=harp_chunk.disk_serial_number,
      longitude=harp_chunk.longitude,
      latitude=harp_chunk.latitude,
      depth=harp_chunk.depth,
      subchunks=harp_chunk_subchunks,
  ).write(output)
  output.write(struct.pack('<4sI', b'data', file_size - data_begin_pos))
  # PCM audio was written earlier, to know Subchunk.byte_loc.
  output.seek(0)


class Reader:
  """Controller for reading XWAV audio one subchunk at a time."""

  def __init__(self, byte_stream: BinaryIO):
    """Initializes this reader for a given byte stream.

    Args:
      byte_stream: File-like object positioned at the beginning of an entire
        XWAV file, uncompressed or FLAC-compressed with --keep-foreign-metadata.

    Raises:
      CorruptHeadersError: if the reader does not read valid WAV file contents.
    MissingChunkError: if the WAV headers are valid but the "harp" or "fmt "
      chunk is missing.
    """
    try:
      self._header = header_from_wav(byte_stream)
    except CorruptHeadersError:
      byte_stream.seek(0)
      self._header = header_from_flac(byte_stream)
    byte_stream.seek(0)
    self._soundfile = soundfile.SoundFile(byte_stream)

  @property
  def header(self):
    return self._header

  @property
  def subchunks(self):
    return self._header.harp_chunk.subchunks

  def __iter__(self) -> Iterable[Tuple[Subchunk, np.ndarray]]:
    """Iterator for subchunk metadata and audio.

    The audio is represented as a list of 16-bit integers obtained from
    interpreting the subchunk bytes as 16-bit little-endian.

    Yields:
      (subchunk, audio) pairs for the entire file.
    """
    for subchunk in self.subchunks:
      yield subchunk, self._read_subchunk(subchunk)

  def __getitem__(self, subchunk_index):
    subchunk = self.subchunks[subchunk_index]
    return subchunk, self._read_subchunk(subchunk)

  def _read_subchunk(self, subchunk: Subchunk) -> np.ndarray:
    """Reads the audio for a given subchunk.

    Args:
      subchunk: A subchunk from the byte stream passed to __init__. It will be
        used to determine the offset and length of the audio read.

    Returns:
      A (frames x channels) numpy array of int16 PCM audio.

    Raises:
      OutOfRangeError: if the data range referenced by the subchunk argument is
        outside the byte stream.
    """
    block_align = self.header.fmt_chunk.block_align
    audio_begin_byte_offset = self.subchunks[0].byte_loc
    frame_offset = (subchunk.byte_loc - audio_begin_byte_offset) // block_align
    subchunk_duration_frames = subchunk.byte_length // block_align

    self._soundfile.seek(frame_offset, whence=soundfile.SEEK_SET)
    samples = self._soundfile.read(frames=subchunk_duration_frames,
                                   dtype='int16',
                                   always_2d=True)

    frames, channels = samples.shape
    del channels
    if subchunk.byte_length != frames * block_align:
      raise OutOfRangeError('subchunk was truncated')
    return samples
