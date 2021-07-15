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
import datetime
import enum
import io
import struct
from typing import BinaryIO, Iterable, List, Sequence, Tuple, TypeVar

from dataclasses import dataclass
import mutagen
from mutagen import flac

FMT_CHUNK_ID = b'fmt '  # (trailing space intentional)
HARP_CHUNK_ID = b'harp'


class Error(Exception):
  """Base class exceptions related to the XWAV format."""
  pass


class NoHarpChunkError(Error):
  """Raised when reading an otherwise valid file with no "harp" chunk."""


class CorruptHeadersError(Error):
  """Raised when XWAV headers are not in the expected format."""
  pass


class OutOfRangeError(Error):
  """Raised when XWAV headers reference an inaccessible file position."""
  pass


class AudioFormatError(Error):
  """Raised in case of an unsupported WAVE audio format code."""
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


@dataclass(frozen=True)
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


# For type annotations of factory methods.
SubchunkType = TypeVar('SubchunkType', bound='Subchunk')
HarpChunkType = TypeVar('HarpChunkType', bound='HarpChunk')


@dataclass(frozen=True)
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
      time = datetime.datetime(cls.ZERO_YEAR + year, month, day, hour, minute,
                               second, ticks * cls.MICROSECONDS_PER_TICK)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class Header:
  """Full parsed XWAV header, including standard WAV headers."""
  fmt_chunk: FmtChunk
  harp_chunk: HarpChunk


def _clean_header_str(value: bytes) -> str:
  """Null-terminates, strips, and removes trailing underscores."""
  return value.split(b'\x00')[0].decode().strip().rstrip('_')


def header_from_wav(reader: BinaryIO) -> Header:
  """Parses an XWAV header from a WAV file.

  Args:
    reader: file-like object positioned at the beginning of an uncompressed XWAV
      file.

  Returns:
    Header dataclass parsed from WAV headers or None if any chunks are missing.

  Raises:
    CorruptHeadersError: if the reader does not read valid WAV file contents.
    NoHarpChunkError: if the WAV headers are valid but there is no "harp" chunk.
  """
  try:
    riff, riff_size, riff_format = struct.unpack('<4sI4s', reader.read(12))
  except struct.error as e:
    raise CorruptHeadersError from e
  if riff != b'RIFF':
    raise CorruptHeadersError('not in RIFF format')
  del riff_size
  if riff_format != b'WAVE':
    raise CorruptHeadersError('RIFF identifier was not WAVE')

  fmt_chunk = None
  harp_chunk = None
  current_chunk = None
  chunk_id = riff
  while chunk_id != HARP_CHUNK_ID:
    if current_chunk:
      current_chunk.skip()
    try:
      current_chunk = chunk.Chunk(reader, bigendian=False)
      chunk_id = current_chunk.getname()
      if chunk_id == FMT_CHUNK_ID:
        fmt_chunk = FmtChunk.read(current_chunk)
      if chunk_id == HARP_CHUNK_ID:
        if not fmt_chunk:
          raise CorruptHeadersError('"fmt " chunk should precede "harp" chunk')
        harp_chunk = HarpChunk.read(reader)
    except EOFError:
      break

  if not (fmt_chunk and harp_chunk):
    raise NoHarpChunkError
  return Header(fmt_chunk=fmt_chunk, harp_chunk=harp_chunk)


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
    NoHarpChunkError: if the FLAC file is valid but there is no "harp" chunk in
      the "riff" APPLICATION metadata.
  """
  application_metadata_code = 2

  try:
    flac_reader = flac.FLAC(reader)
  except mutagen.MutagenError as e:
    raise CorruptHeadersError from e

  fmt_chunk = None
  harp_chunk = None
  for metadata_block in flac_reader.metadata_blocks:
    if metadata_block.code == application_metadata_code:
      try:
        riff, chunk_id, size = struct.unpack('<4s4sI', metadata_block.data[:12])
      except struct.error as e:
        raise CorruptHeadersError from e
      if riff != b'riff':
        raise CorruptHeadersError(
            'expected FLAC APPLICATION metadata to be of type "riff"')
      del size
      chunk_reader = io.BytesIO(metadata_block.data[12:])
      if chunk_id == FMT_CHUNK_ID:
        fmt_chunk = FmtChunk.read(chunk_reader)
      if chunk_id == HARP_CHUNK_ID:
        harp_chunk = HarpChunk.read(chunk_reader)

  if not (fmt_chunk and harp_chunk):
    raise NoHarpChunkError
  return Header(fmt_chunk=fmt_chunk, harp_chunk=harp_chunk)


class Reader:
  """Controller for reading XWAV audio one subchunk at a time."""

  # All real XWAVs known to the authors are 16-bit integer PCM.
  # Treating this as a constant and validating against headers simplifies the
  # implementation.
  BYTES_PER_SAMPLE = 2

  def __init__(self, byte_stream: BinaryIO):
    """Initializes this reader for a given byte stream.

    Args:
      byte_stream: File-like object positioned at the beginning of an entire
        XWAV file.

    Raises:
      CorruptHeadersError: if the reader does not read valid WAV file contents.
      NoHarpChunkError: if the WAV headers are valid but there is no "harp"
        chunk.
      AudioFormatError: if the WAVE audio format code is unsupported.
    """
    self._byte_stream = byte_stream
    self._header = header_from_wav(byte_stream)

    fmt_chunk = self._header.fmt_chunk
    audio_format = fmt_chunk.audio_format
    if audio_format != AudioFormat.PCM:
      raise AudioFormatError('only PCM format is supported; got ' +
                             audio_format)
    if fmt_chunk.block_align != self.BYTES_PER_SAMPLE * fmt_chunk.num_channels:
      raise AudioFormatError('only 16-bit integer PCM is supported')

  @property
  def header(self):
    return self._header

  @property
  def subchunks(self):
    return self._header.harp_chunk.subchunks

  def __iter__(self) -> Iterable[Tuple[Subchunk, List[int]]]:
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

  def _read_subchunk(self, subchunk: Subchunk) -> List[int]:
    self._byte_stream.seek(subchunk.byte_loc)
    read_bytes = self._byte_stream.read(subchunk.byte_length)
    if len(read_bytes) < subchunk.byte_length:
      raise OutOfRangeError
    num_samples = subchunk.byte_length // self.BYTES_PER_SAMPLE
    return struct.unpack('<%dh' % num_samples, read_bytes)
