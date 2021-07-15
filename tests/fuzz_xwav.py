"""Fuzz test runner for XWAV header reading.

See https://github.com/google/atheris for installation instructions for the
fuzzing enging. It isn't included in setup.py, since it is unly useful during
development and it's heavy enough to be worth not building during initialization
of, for example, Dataflow workers that depend on this package.
"""

import io
import sys

import atheris

with atheris.instrument_imports():
  from multispecies_whale_detection import xwav


def TestParseHarpChunk(data):
  try:
    xwav.HarpChunk.read(io.BytesIO(data))
  except xwav.CorruptHeadersError:
    pass


if __name__ == '__main__':
  atheris.Setup(sys.argv, TestParseHarpChunk)
  atheris.Fuzz()
