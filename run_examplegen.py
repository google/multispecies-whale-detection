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
"""Runner and configuation for the examplegen Beam pipeline.

Usage:
  GOOGLE_APPLICATION_CREDENTIALS=service_account.json python3 run_examplegen.py

As configured, this runs on Google Cloud Dataflow with inputs and outputs in
Google Cloud Storage. Users need to edit at least:
  * input_directory
  * output_directory
  * temp_location
  * project

And possibly:
  * region, if input/output/temp are in a single-region bucket.
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
      resample_rate=4000,
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
