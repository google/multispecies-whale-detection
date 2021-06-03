"""Runner and configuation for the examplegen Beam pipeline.

Usage:
  GOOGLE_APPLICATION_CREDENTIALS=service_account.json python3 run_examplegen.py

As configured, this runs on Google Cloud Dataflow with inputs and outputs in
Google Cloud Storage. Users need to edit at least:
  * input_directory
  * output_directory
  * temp_location
  * project
"""
import os
import pathlib

from apache_beam.options import pipeline_options

from multispecies_whale_detection import examplegen


def run():
  """Configures and runs the pipeline and waits on completion."""
  configuration = examplegen.Configuration(
      input_directory='gs://msw-dev/input',
      output_directory='gs://msw-dev/output',
      resample_rate=16000,
      clip_duration_seconds=10.0,
  )

  options = pipeline_options.PipelineOptions(
      temp_location='gs://msw-dev/tmp',
      project='bioacoustics-216319',
      region='us-central1',
      job_name='examplegen',
      runner='DataflowRunner',
      setup_file=os.path.join(pathlib.Path(__file__).parent, 'setup.py'),
  )

  examplegen.run(configuration, options)


if __name__ == '__main__':
  run()
