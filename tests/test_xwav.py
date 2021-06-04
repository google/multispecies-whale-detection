import datetime
import io
import os
import unittest

from multispecies_whale_detection import xwav


def fixture_path(basename: str) -> str:
  return os.path.join(os.path.dirname(__file__), basename)


def fixture_header():
  return xwav.Header(
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
              time=datetime.datetime(2010, 2, 28, 2, 21, 15),
              byte_loc=34444,
              byte_length=1500000,
              write_length=3000,
              sample_rate=10000,
              gain=1),
          xwav.Subchunk(
              time=datetime.datetime(2010, 2, 28, 2, 22, 30),
              byte_loc=1534444,
              byte_length=1500000,
              write_length=3000,
              sample_rate=10000,
              gain=1)
      ])


class TestXwav(unittest.TestCase):

  def test_header_from_wav(self):
    with open(fixture_path('xwav_headers_only.x.wav'), 'rb') as reader:
      header = xwav.header_from_wav(reader)
      self.assertEqual(fixture_header(), header)

  def test_header_from_flac(self):
    with open(fixture_path('xwav_headers_only.x.flac'), 'rb') as reader:
      header = xwav.header_from_flac(reader)
      self.assertEqual(fixture_header(), header)

  def test_plain_wav_header_is_none(self):
    with open(fixture_path('plain_headers.wav'), 'rb') as reader:
      header = xwav.header_from_wav(reader)
      self.assertIsNone(header)

  def test_plain_flac_header_is_none(self):
    with open(fixture_path('plain_headers.flac'), 'rb') as reader:
      header = xwav.header_from_flac(reader)
      self.assertIsNone(header)

  def test_empty_wav_raises(self):
    with self.assertRaises(Exception):
      _ = xwav.header_from_wav(io.BytesIO())

  def test_corrupt_wav_raises(self):
    with self.assertRaises(Exception):
      _ = xwav.header_from_wav(io.BytesIO(b'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'))

  def test_corrupt_flac_raises(self):
    with self.assertRaises(Exception):
      _ = xwav.header_from_flac(io.BytesIO(b'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'))


if __name__ == '__main__':
  unittest.main()
