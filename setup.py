"""multispecies_whale_detection package specification.

This is used both to provide a pip install for command line utilities that
enable preprocessing and model training and to initialize Beam workers for the
examplegen pipeline.
"""
# pylint: disable=g-importing-member
from distutils.command.build import build as _build  # type: ignore
# pylint: enable=g-importing-member
import subprocess

import setuptools


class build(_build):  # pylint: disable=invalid-name
  """A build command class that will be invoked during package install.

  The package built using the current setup.py will be staged and later
  installed in the worker using `pip install package'. This class will be
  instantiated during install for this specific scenario and will trigger
  running the custom commands specified.
  """
  sub_commands = _build.sub_commands + [('CustomCommands', None)]


CUSTOM_COMMANDS = [
    ['apt-get', 'update'],
    ['apt-get', '--assume-yes', 'install', 'libsndfile1'],
]


class CustomCommands(setuptools.Command):
  """A setuptools Command class able to run arbitrary commands."""

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def RunCustomCommand(self, command_list):
    p = subprocess.Popen(
        command_list,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    stdout_data, _ = p.communicate()
    if p.returncode != 0:
      raise RuntimeError('Command %s failed: exit code: %s' %
                         (command_list, p.returncode))

  def run(self):
    for command in CUSTOM_COMMANDS:
      self.RunCustomCommand(command)


setuptools.setup(
    name='multispecies_whale_detection',
    version='0.0.1',
    url='https://github.com/google/multispecies_whale_detection',
    author='Google Bioacoustics Project',
    author_email='bioacoustics-project@google.com',
    install_requires=[
        'apache_beam[gcp]',
        'python-dateutil',
        'intervaltree',
        'mutagen',
        'numpy',
        'python-snappy',
        'resampy',
        'soundfile',
        'tensorflow',
    ],
    packages=setuptools.find_packages(),
    cmdclass={
        'build': build,
        'CustomCommands': CustomCommands,
    },
)
