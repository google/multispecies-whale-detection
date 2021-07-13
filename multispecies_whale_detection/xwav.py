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
import io
import struct
from typing import BinaryIO, Optional, Sequence, TypeVar
import wave

from dataclasses import dataclass
from mutagen import flac

FMT_CHUNK_ID = b'fmt '  # (trailing space intentional)
HARP_CHUNK_ID = b'harp'


@dataclass(frozen=True)
class FmtChunk:
  """Metadata from the standard WAV "fmt " chunk."""
  num_channels: int
  sample_rate: int
  bytes_per_second: int
  block_align: int
  bits_per_sample: int


@dataclass(frozen=True)
class Subchunk:
  """HARP subchunk metadata, including length, position, and UTC start time."""
  time: datetime
  byte_loc: int
  byte_length: int
  write_length: int
  sample_rate: int
  gain: int


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


# to annotate a factory method
HeaderType = TypeVar('HeaderType', bound='Header')


@dataclass(frozen=True)
class Header:
  """Full parsed XWAV header, including standard WAV headers."""
  fmt_chunk: FmtChunk
  harp_chunk: HarpChunk

  @classmethod
  def from_chunks(cls, fmt_chunk: Optional[FmtChunk],
                  harp_chunk: Optional[HarpChunk]) -> Optional[HeaderType]:
    """Constructs a header only if both arguments are not None."""
    if fmt_chunk is None or harp_chunk is None:
      return None
    return cls(fmt_chunk, harp_chunk)


def _clean_header_str(value: bytes) -> str:
  """Null-terminates, strips, and removes trailing underscores."""
  return value.split(b'\x00')[0].decode().strip().rstrip('_')


def read_fmt_chunk(reader: BinaryIO) -> FmtChunk:
  """Parses a WAV "fmt " chunk.

  Args:
    reader: file-like object that reads the contents of a "fmt " chunk. (It
      should be positioned at the start of the content, that is after the size,
      of the "fmt " chunk.)

  Returns:
    FmtChunk parsed from the reader.
  """
  (format_code, num_channels, sample_rate, bytes_per_second, block_align,
   bits_per_sample) = struct.unpack_from('<HHLLHH', reader.read(16))
  del format_code
  return FmtChunk(
      num_channels=num_channels,
      sample_rate=sample_rate,
      bytes_per_second=bytes_per_second,
      block_align=block_align,
      bits_per_sample=bits_per_sample,
  )


def read_harp_chunk(reader: BinaryIO) -> HarpChunk:
  """Parses an XWAV header with its associated subchunks.

  Args:
    reader: file-like object that reads the contents of a "harp" chunk. (In the
      context of the WAV (RIFF) file format, it should be positioned at the
      start of the content, that is after the size, of the HARP chunk.)

  Returns:
    HarpChunk dataclass parsed from the reader.
  """

  geo_scale = 100000  # decimal degrees per unit of int latitude and longitude
  microseconds_per_tick = 1000  # HARP "ticks" time scale

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
  latitude = latitude / geo_scale
  longitude = longitude / geo_scale

  subchunks = []
  for _ in range(num_raw_files):
    year, month, day, hour, minute, second, ticks = struct.unpack(
        '<BBBBBBH', reader.read(8))
    byte_loc, byte_length, write_length, sample_rate = struct.unpack(
        '<IIII', reader.read(16))
    gain = struct.unpack('B7x', reader.read(8))[0]

    subchunks.append(
        Subchunk(
            time=datetime.datetime(2000 + year, month, day, hour, minute,
                                   second, ticks * microseconds_per_tick),
            byte_loc=byte_loc,
            byte_length=byte_length,
            write_length=write_length,
            sample_rate=sample_rate,
            gain=gain,
        ))

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


def header_from_wav(reader: BinaryIO) -> Optional[Header]:
  """Parses an XWAV header from a WAV file.

  Args:
    reader: file-like object positioned at the beginning of an uncompressed XWAV
      file.

  Returns:
    Header dataclass parsed from WAV headers or None if any chunks are missing.

  Raises:
    ValueError if the reader does not read valid WAV file contents.
  """
  riff, riff_size, riff_format = struct.unpack('<4sI4s', reader.read(12))
  if riff != b'RIFF':
    raise ValueError('not in RIFF format')
  del riff_size
  if riff_format != b'WAVE':
    raise ValueError('not a WAVE file')

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
        fmt_chunk = read_fmt_chunk(current_chunk)
      if chunk_id == HARP_CHUNK_ID:
        if not fmt_chunk:
          raise ValueError('"fmt " chunk should precede "harp" chunk')
        harp_chunk = read_harp_chunk(reader)
    except EOFError:
      return None

  return Header.from_chunks(fmt_chunk, harp_chunk)


def header_from_flac(reader: BinaryIO) -> Optional[Header]:
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
  """
  application_metadata_code = 2

  flac_reader = flac.FLAC(reader)

  fmt_chunk = None
  harp_chunk = None
  for metadata_block in flac_reader.metadata_blocks:
    if metadata_block.code == application_metadata_code:
      riff, chunk_id, size = struct.unpack('<4s4sI', metadata_block.data[:12])
      assert riff == b'riff'
      del size
      chunk_reader = io.BytesIO(metadata_block.data[12:])
      if chunk_id == FMT_CHUNK_ID:
        fmt_chunk = read_fmt_chunk(chunk_reader)
      if chunk_id == HARP_CHUNK_ID:
        harp_chunk = read_harp_chunk(chunk_reader)

  return Header.from_chunks(fmt_chunk, harp_chunk)
