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
