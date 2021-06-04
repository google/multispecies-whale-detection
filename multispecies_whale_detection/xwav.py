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
from typing import BinaryIO, Optional, Sequence

from dataclasses import dataclass
from mutagen import flac

HARP_CHUNK_ID = b'harp'


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
class Header:
  """File-level XWAV metadata."""
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


def _clean_header_str(value: bytes) -> str:
  """Null-terminates, strips, and removes trailing underscores."""
  return value.split(b'\x00')[0].decode().strip().rstrip('_')


def header_from_harp_chunk(reader: BinaryIO) -> Header:
  """Parses an XWAV header with its associated subchunks.

  Args:
    reader: file-like object that reads the contents of a "harp" chunk. (In the
      context of the WAV (RIFF) file format, it should be positioned at the
      start of the content, that is after the size, of the HARP chunk.)

  Returns:
    Header dataclass parsed from the given HARP chunk.
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

  header = Header(
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

  return header


def header_from_wav(reader: BinaryIO) -> Optional[Header]:
  """Parses an XWAV header from a WAV file.

  Args:
    reader: file-like object positioned at the beginning of an uncompressed XWAV
      file.

  Returns:
    Header dataclass parsed from the "harp" chunk or None if there is no such
    chunk.

  Raises:
    ValueError if the reader does not read valid WAV file contents.
  """
  riff, riff_size, riff_format = struct.unpack('<4sI4s', reader.read(12))
  if riff != b'RIFF':
    raise ValueError('not in RIFF format')
  del riff_size
  if riff_format != b'WAVE':
    raise ValueError('not a WAVE file')

  current_chunk = None
  chunk_id = riff
  while chunk_id != HARP_CHUNK_ID:
    if current_chunk:
      current_chunk.skip()
    try:
      current_chunk = chunk.Chunk(reader, bigendian=False)
      chunk_id = current_chunk.getname()
    except EOFError:
      return None

  if chunk_id == HARP_CHUNK_ID:
    return header_from_harp_chunk(reader)
  else:
    return None


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
    Header dataclass parsed from the "harp" chunk.
  """
  application_metadata_code = 2

  flac_reader = flac.FLAC(reader)
  for metadata_block in flac_reader.metadata_blocks:
    if metadata_block.code == application_metadata_code:
      riff, chunk_id, size = struct.unpack('<4s4sI', metadata_block.data[:12])
      del size
      assert riff == b'riff'
      if chunk_id == HARP_CHUNK_ID:
        return header_from_harp_chunk(io.BytesIO(metadata_block.data[12:]))
  return None
