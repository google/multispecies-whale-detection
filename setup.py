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
"""multispecies_whale_detection package specification.

Purposes:
  * Configuration for pip install (locally and in Dataflow)
  * apt install of libsndfile in Dataflow workers.

Usage:
  pip install .  # In the directory containing this file

For local usage this trys to install libsndfile1, giving up if there is no apt
command (such as on Windows or OS X, where PySoundFile installs its own
dependencies) or if the user has insufficient privilege to run apt-get (in which
case it is recommended to apt-get install libsndfile1 manually).
"""
import ctypes
# pylint: disable=g-importing-member
from distutils.command.build import build as _build  # type: ignore
# pylint: enable=g-importing-member
import subprocess

import setuptools


class build(_build):  # pylint: disable=invalid-name
  """Override of the build command to install libsndfile when appropriate."""
  sub_commands = _build.sub_commands + [('maybe_install_libsndfile', None)]


def _can_call_apt():
  return subprocess.run(['apt-get', 'help'], check=False).returncode == 0


def _sndfile_installed():
  try:
    ctypes.cdll.LoadLibrary('libsndfile.so.1')
    return True
  except OSError:
    return False


class MaybeInstallLibsndfile(setuptools.Command):
  """A setuptools Command class that installs libsndfile on Linux with apt."""

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def run(self):
    if not _can_call_apt():
      # Assuming we are on Windows or OS X where PySoundFile will manually
      # install its C dependency.
      return
    if _sndfile_installed():
      # Woo-hoo. We are derived from Debian, and PySoundFile will already be
      # happy to have its dependency.
      return
    try:
      subprocess.run(['apt-get', 'update'], check=True)
      subprocess.run(['apt-get', '--assume-yes', 'install', 'libsndfile1'],
                     check=True)
    except subprocess.CalledProcessError:
      raise Exception(
          'We tried to call apt-get but it failed. We recommend installing '
          'manually, like "sudo apt-get install libsndfile1" and re-running '
          'pip install.')


setuptools.setup(
    name='multispecies_whale_detection',
    version='0.0.1',
    url='https://github.com/google/multispecies_whale_detection',
    author='Google Bioacoustics Project',
    author_email='bioacoustics-project@google.com',
    install_requires=[
        'apache_beam',
        'apache_beam[gcp]',
        'python-dateutil',
        'intervaltree',
        'mutagen',
        'numpy',
        'resampy',
        'soundfile',
        'tensorflow',
        'tensorflow_probability',
        # not a direct dependency, but beam.io.tfrecordio asks for this to be as
        # "fast as it could be"
        'python-snappy',
    ],
    packages=['multispecies_whale_detection'],
    cmdclass={
        'build': build,
        'maybe_install_libsndfile': MaybeInstallLibsndfile,
    },
)
