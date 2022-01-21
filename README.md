# Multispecies Whale Detection

This repository contains tools Google implemented for the development of a
neural network to detect whale vocalizations and classify them across several
species and geographies. We release the code in the hope that it will allow
other groups to more easily develop species detection models on their own data.

This is not an officially supported Google product. Support and/or new releases
may be limited.

## Example Generator (examplegen)

A Beam pipeline for creating trainer input from a collection of labeled audio
files.

### Motivation

Acoustic monitoring datasets are normally stored as a collection of audio files
on a filesystem. The duration of the files varies across and within deployments
but is often much longer than the ideal length of individual training examples.
Sample rate, audio encoding, and number of channels also vary, while the trainer
will require a single input format for audio and annotations.  Annotations,
which are the source of labeled data for training, are usually stored as CSV,
but the naming and meaning of columns also varies.

### Features

The *examplegen* Beam pipeline reads audio files and CSV label and metadata files
from a filesystem and writes TFRecord files to be consumed by the training job.
In the process, it:

*   handles different audio formats
*   splits multi-channel files and resamples to a common sample rate
*   chunks large files into clips short enough to not slow down training
*   joins labels and metadata to the audio, and represents them as features in
    the output TensorFlow Examples
*   adjusts label start times to be relative to the start of the audio clip in
    each example
*   serializes the joined records in tensorflow.Example format

### Usage

Local:
1.  Run `python3 run_examplegen_local.py`

Google Cloud Dataflow:

1.  Edit run\_examplegen.py, setting the project and bucket paths for your
    own Cloud project or switching to a different Beam runner.
2.  In your Cloud project, create a service account and IAM permissions for
    "Dataflow Admin" and "Storage Object Admin" on the bucket paths you
    configured in run_examplegen.py.
4.  Generate a service account key, download the JSON file, and rename it to
    service-account.json.
3.  Run `GOOGLE_APPLICATION_CREDENTIALS=service_account.json python3 run_examplegen.py`.
4.  Monitor the job and, when it completes, check the output in the
    `output_directory` you configured.

## Audio TFRecord inspector (dataset.print)

A debugging tool for inspecting the output of the examplegen pipeline.

Usage:

```
python3 -m multispecies_whale_detection.scripts.print_dataset --tfrecord_filepattern={output_directory}/tfrecords-*
```

This prints to the console a human-readable representation of the first few
records of an audio TFRecord dataset specified by the given file pattern. It is
mostly intended to be used by developers of this project, but it can be handy to
verify that a run of examplegen produced the expected form of output.

## Dataflow Python versions and virtualenv

(documentation sketch) Python versions used on local machines tend to get
ahead of the versions supported in Dataflow. To avoid unforseen issues, it
is best to launch the Dataflow job from a local machine with a
Dataflow-supported version. virtualenv is a good way to do this alongside any
other Python version that may be installed system-wide.

1.  Determine the latest Python version supported by Dataflow.
2.  Download and install that Python version.
3.  Create and activate the virtual environment.
    ```
    python3 -m venv ~/tmp/whale_env
    . ~/tmp/whale_env/bin/activate
    pip install --upgrade pip
    ```
4.  Install this package in the virtual environment.
    ```
    cd multispecies_whale_detection  # (directory containing setup.py)
    pip install .
    ```
