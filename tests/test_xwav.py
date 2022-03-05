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
import io
import os
import shutil
import struct
from typing import BinaryIO
import unittest
import wave

from multispecies_whale_detection import xwav

# WAVE file chunks start with a 4-byte chunk identifier, followed by a 32-bit
# unsigned integer size.
CHUNK_PREAMBLE_LENGTH = 8

HARP_CHUNK_ID = b'harp'


def fixture_path(basename: str) -> str:
  return os.path.join(os.path.dirname(__file__), basename)


def fixture_header():
  """Returns the expected parsed contents of the fixture HARP header.

  (This is the start of one XWAV file from a real deployment, but the audio is
  not included, since real XWAV files are typically much too large to live in
  source control.)
  """
  return xwav.Header(
      fmt_chunk=xwav.FmtChunk(
          audio_format=xwav.AudioFormat.PCM,
          num_channels=1,
          sample_rate=10000,
          bytes_per_second=20000,
          block_align=2,
          bits_per_sample=16,
      ),
      harp_chunk=xwav.HarpChunk(
          wav_version_number=1,
          firmware_version_number='V2.01A',
          instrument_id='DL41',
          site_name='',
          experiment_name='Kauai01',
          disk_sequence_number=11,
          disk_serial_number='12345678',
          longitude=-159.53383,
          latitude=21.57224,
          depth=720,
          subchunks=[
              xwav.Subchunk(
                  time=datetime.datetime(
                      2010, 2, 28, 2, 21, 15, tzinfo=datetime.timezone.utc),
                  byte_loc=34444,
                  byte_length=1500000,
                  write_length=3000,
                  sample_rate=10000,
                  gain=1),
              xwav.Subchunk(
                  time=datetime.datetime(
                      2010, 2, 28, 2, 22, 30, tzinfo=datetime.timezone.utc),
                  byte_loc=1534444,
                  byte_length=1500000,
                  write_length=3000,
                  sample_rate=10000,
                  gain=1)
          ]),
  )


def fixture_two_chunk_plain_wav() -> BinaryIO:
  """Creates a fixture WAVE file with two distinct sections.

  The audio is 100Hz mono. Each section 10 samples long. Samples in the first
  alternate between +/-(1 << 5) and in the second between +/-(1 << 10).

  Returns:
    File-like object with the bytes of the fixture WAVE file, positioned at the
      beginning.
  """
  sample_rate = 100
  chunk_duration_samples = 10

  plain_wav_io = io.BytesIO()
  with wave.open(plain_wav_io, 'wb') as writer:
    writer.setnchannels(1)
    writer.setsampwidth(2)
    writer.setframerate(sample_rate)
    signs = [pow(-1, i) for i in range(chunk_duration_samples)]
    for magnitude in [(1 << 5), (1 << 10)]:
      writer.writeframes(
          struct.pack('<%dh' % len(signs), *[magnitude * s for s in signs]))
  plain_wav_io.seek(0)
  return plain_wav_io


def wav_data_start(reader: BinaryIO) -> int:
  """Returns the byte position of the first audio sample in a WAVE file."""
  pos = reader.tell()
  reader.seek(0)
  with wave.open(reader, 'rb') as wav_reader:
    wav_reader.rewind()
    data_start = reader.tell()
  reader.seek(pos)
  return data_start


def fixture_two_chunk_harp_chunk(data_start: int) -> xwav.HarpChunk:
  """Returns a HarpChunk describing fixture_two_chunk_plain_wav.

  This has two Subchunks, corresponding to the "sections" described in the
  docstring of fixture_two_chunk_plain_wav. Their start times and the rest of
  the metadata are valid but have arbitrary values.

  Args:
    data_start: Index of the first byte of audio in the plain WAV.

  Returns:
    HarpChunk corresponding to the "two_chunk" fixture.
  """

  def round_coord(c: float) -> float:
    """Rounds a float to the precision of XWAV's int lat/long."""
    return round(c * xwav.HarpChunk.GEO_SCALE) / xwav.HarpChunk.GEO_SCALE

  return xwav.HarpChunk(
      wav_version_number=1,
      firmware_version_number='V1.0',
      instrument_id='FAKE',
      site_name='abcd',
      experiment_name='rewrite',
      disk_sequence_number=1,
      disk_serial_number='X123',
      longitude=round_coord(-122.391562122),
      latitude=round_coord(37.791205),
      depth=6,
      subchunks=[
          xwav.Subchunk(
              time=datetime.datetime(
                  2021, 7, 15, 0, 27, 00, tzinfo=datetime.timezone.utc),
              byte_loc=data_start + 0,
              byte_length=20,
              write_length=20,
              sample_rate=100,
              gain=1),
          xwav.Subchunk(
              time=datetime.datetime(
                  2021, 7, 15, 0, 28, 00, tzinfo=datetime.timezone.utc),
              byte_loc=data_start + 20,
              byte_length=20,
              write_length=20,
              sample_rate=100,
              gain=1)
      ],
  )


def insert_harp_chunk(harp_chunk: xwav.HarpChunk,
                      wav_reader: BinaryIO) -> BinaryIO:
  """Inserts a HarpChunk into the headers of a WAVE file.

  Args:
    harp_chunk: The HarpChunk to insert.
    wav_reader: File-like object positioned at the start of a WAVE file.

  Returns:
    File-like object positioned at the beginning of an XWAV file for which
      harp_chunk has been serialized into a "harp" chunk between the "fmt " and
      "data" chunks.
  """
  # The implementation starts with the bytes of the plain WAVE fixture and
  # populates a new BytesIO, in order, with knowledge of the details of the WAVE
  # file format, for which a useful reference is
  #
  # http://soundfile.sapp.org/doc/WaveFormat/

  xwav_io = io.BytesIO()

  harp_chunk_io = io.BytesIO()
  harp_chunk.write(harp_chunk_io)
  serialized_harp_chunk = harp_chunk_io.getbuffer()

  # Rewrite the entire file size.
  harp_id_and_size = struct.pack('<4sI', HARP_CHUNK_ID,
                                 len(serialized_harp_chunk))
  entire_file_size = (
      len(wav_reader.getbuffer()) + len(serialized_harp_chunk) +
      len(harp_id_and_size))
  riff_chunk = struct.pack('<4sI4s', b'RIFF',
                           entire_file_size - CHUNK_PREAMBLE_LENGTH, b'WAVE')

  data_chunk_start = wav_data_start(wav_reader) - CHUNK_PREAMBLE_LENGTH
  wav_reader.seek(len(riff_chunk))
  fmt_chunk = wav_reader.read(data_chunk_start - len(riff_chunk))

  xwav_io.write(riff_chunk)
  xwav_io.write(fmt_chunk)

  # "harp" chunk
  xwav_io.write(harp_id_and_size)
  xwav_io.write(serialized_harp_chunk)

  shutil.copyfileobj(wav_reader, xwav_io)

  xwav_io.seek(0)
  return xwav_io


def fixture_two_chunk_xwav() -> BinaryIO:
  """Adds a HARP chunk to the headers of fixture_two_chunk_plain_wav.

  This fixture enables testing subchunk-at-a-time reads and seeks to specified
  subchunks.

  Returns:
    File-like object with the bytes of the fixture XWAV file, positioned at the
      beginning.
  """
  plain_wav_io = fixture_two_chunk_plain_wav()
  data_start_no_harp = wav_data_start(plain_wav_io)

  # The eventual size of the HARP chunk will be required for computing
  # correct Subchunk.byte_loc values.
  harp_chunk_size = xwav.HarpChunk.serialized_len(num_subchunks=2)
  harp_id_and_size = struct.pack('<4sI', HARP_CHUNK_ID, harp_chunk_size)
  data_start = data_start_no_harp + len(harp_id_and_size) + harp_chunk_size

  return insert_harp_chunk(
      fixture_two_chunk_harp_chunk(data_start=data_start), plain_wav_io)


class TestXwav(unittest.TestCase):

  def test_header_from_wav(self):
    with open(fixture_path('xwav_headers_only.x.wav'), 'rb') as reader:
      header = xwav.header_from_wav(reader)
      self.assertEqual(fixture_header(), header)

  def test_read_empty_harp_chunk(self):
    with self.assertRaises(xwav.CorruptHeadersError):
      xwav.HarpChunk.read(io.BytesIO())

  def test_read_write_read_harp_chunk(self):
    # Since xwav_headers_only.x.wav has a "harp" chunk that was written out by
    # Triton, a round trip through the library under test provides additional
    # verification.
    with open(fixture_path('xwav_headers_only.x.wav'), 'rb') as reader:
      header = xwav.header_from_wav(reader)
    harp_chunk = header.harp_chunk

    rewritten_io = io.BytesIO()
    harp_chunk.write(rewritten_io)
    rewritten_io.seek(0)

    reread_harp_chunk = xwav.HarpChunk.read(rewritten_io)
    self.assertEqual(harp_chunk, reread_harp_chunk)

  def test_header_from_flac(self):
    with open(fixture_path('xwav_headers_only.x.flac'), 'rb') as reader:
      header = xwav.header_from_flac(reader)
      self.assertEqual(fixture_header(), header)

  def test_plain_wav_header_is_none(self):
    with self.assertRaises(xwav.MissingChunkError):
      with open(fixture_path('plain_headers.wav'), 'rb') as reader:
        _ = xwav.header_from_wav(reader)

  def test_plain_flac_header_is_none(self):
    with self.assertRaises(xwav.MissingChunkError):
      with open(fixture_path('plain_headers.flac'), 'rb') as reader:
        _ = xwav.header_from_flac(reader)

  def test_empty_wav_raises(self):
    with self.assertRaises(xwav.CorruptHeadersError):
      _ = xwav.header_from_wav(io.BytesIO())

  def test_corrupt_wav_raises(self):
    with self.assertRaises(xwav.CorruptHeadersError):
      _ = xwav.header_from_wav(io.BytesIO(b'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'))

  def test_corrupt_flac_raises(self):
    with self.assertRaises(xwav.CorruptHeadersError):
      _ = xwav.header_from_flac(io.BytesIO(b'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'))

  def test_initialize_wav_reader(self):
    with open(fixture_path('xwav_headers_only.x.wav'), 'rb') as file_reader:
      reader = xwav.Reader(file_reader)
      self.assertIsNotNone(reader.header)

  def test_reader_harp_chunk(self):
    xwav_io: BinaryIO = fixture_two_chunk_xwav()
    fixture_harp_chunk = fixture_two_chunk_harp_chunk(
        data_start=wav_data_start(xwav_io))

    reader = xwav.Reader(xwav_io)
    reader_harp_chunk = reader.header.harp_chunk

    self.assertEqual(fixture_harp_chunk, reader_harp_chunk)

  def reader_reads_two_chunk_fixture_template(self, reader):
    """Test template for WAV and FLAC Reader success cases.

    Args:
      reader: A newly-initialized Reader for fixture_two_chunk_xwav.
    """
    iter_subchunks = iter(reader)
    first_subchunk, first_samples = next(iter_subchunks)
    second_subchunk, second_samples = next(iter_subchunks)
    with self.assertRaises(StopIteration):
      next(iter_subchunks)

    # Non-comprehensive validation of values most likely to be interesting.
    # To understand the hard-coded constants, check against the implementation
    # of fixture_two_chunk_plain_wav.
    self.assertEqual(20, first_subchunk.byte_length)
    self.assertEqual(20, second_subchunk.byte_length)
    self.assertEqual(second_subchunk.byte_loc,
                     first_subchunk.byte_loc + first_subchunk.byte_length)
    for sample in first_samples:
      self.assertEqual(1 << 5, abs(sample))
    for sample in second_samples:
      self.assertEqual(1 << 10, abs(sample))

  def test_reader_reads_two_chunk_fixture(self):
    xwav_io: BinaryIO = fixture_two_chunk_xwav()
    reader = xwav.Reader(xwav_io)
    self.reader_reads_two_chunk_fixture_template(reader)

  def test_round_trip_read_write_read(self):
    xwav_io: BinaryIO = fixture_two_chunk_xwav()
    reader = xwav.Reader(xwav_io)
    fmt_chunk = reader.header.fmt_chunk
    harp_chunk = reader.header.harp_chunk
    subchunk_generator = (
        (subchunk.time, samples) for subchunk, samples in reader)

    write_output = io.BytesIO()
    xwav.write(
        num_channels=fmt_chunk.num_channels,
        sample_rate=fmt_chunk.sample_rate,
        harp_chunk=harp_chunk,
        subchunks=subchunk_generator,
        subchunks_len=len(harp_chunk.subchunks),
        output=write_output,
    )

    write_output.seek(0)
    rereader = xwav.Reader(write_output)
    self.reader_reads_two_chunk_fixture_template(rereader)

  def test_read_out_of_range(self):
    # The subchunks in this fixture file actually reference audio beyond the end
    # of the file.
    with open(fixture_path('xwav_headers_only.x.wav'), 'rb') as file_reader:
      reader = xwav.Reader(file_reader)
      with self.assertRaises(xwav.OutOfRangeError):
        next(iter(reader))

  def test_read_by_index(self):
    xwav_io: BinaryIO = fixture_two_chunk_xwav()
    subchunk_index = 1  # aritrarily the second subchunk of the fixture

    reader = xwav.Reader(xwav_io)
    subchunk, samples = reader[subchunk_index]

    self.assertEqual(10, len(samples))
    self.assertEqual(subchunk, reader.subchunks[subchunk_index])

  def test_initialize_reader_flac(self):
    with open(fixture_path('fixture_two_chunk_xwav.x.flac'), 'rb') as infile:
      reader = xwav.Reader(infile)

      self.assertIsNotNone(reader.header)

  def test_reader_reads_two_chunk_fixture_flac(self):
    with open(fixture_path('fixture_two_chunk_xwav.x.flac'), 'rb') as infile:
      reader = xwav.Reader(infile)

      self.reader_reads_two_chunk_fixture_template(reader)

  def test_reader_reads_from_bytesio_flac(self):
    flac_io = io.BytesIO()
    with open(fixture_path('fixture_two_chunk_xwav.x.flac'), 'rb') as infile:
      flac_io.write(infile.read())
    flac_io.seek(0)
    reader = xwav.Reader(flac_io)

    self.reader_reads_two_chunk_fixture_template(reader)


if __name__ == '__main__':
  unittest.main()
