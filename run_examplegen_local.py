"""Runner and configuation for the examplegen Beam pipeline.

Usage:
  python3 run_examplegen_local.py

The configuration hard-codes subdirectories of the user's home directory for
input and output. This file is intended as configuration-only, as an alternative
to a long command line. Configuration changes should be made by editing it
directly.
"""
import os
import pathlib
import shutil

from apache_beam.options import pipeline_options

from multispecies_whale_detection import examplegen


def run():
  """Configures and runs the pipeline and waits on completion."""
  out_dir = os.path.expanduser('~/tmp/whale/examplegen_output')
  shutil.rmtree(out_dir)
  os.mkdir(out_dir)

  configuration = examplegen.Configuration(
      input_directory=os.path.expanduser('~/tmp/whale/examplegen_input_utc'),
      output_directory=out_dir,
      resample_rate=4000,
      clip_duration_seconds=30.0,
  )

  options = pipeline_options.PipelineOptions(runner='DirectRunner',)

  examplegen.run(configuration, options)


if __name__ == '__main__':
  run()
