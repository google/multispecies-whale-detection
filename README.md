# Multispecies Whale Detection

This repository contains tools Google implemented for the development of a
neural network to detect whale vocalizations and classify them across several
species and geographies. We release the code in the hope that it will allow
other groups to more easily develop species detection models on their own data.

## Example Generator (examplegen)

A Beam pipeline for creating trainer input from a collection of labeled audio
files.

Usage:

1.  Edit examplegen.py, setting the project and bucket paths for your own Cloud
    project or switching to a different Beam runner.
2.  Run `python3 examplegen.py`.
3.  Monitor the job and, when it completes, check the output in the
    `output_directory` you configured.

Acoustic monitoring datasets are normally stored as a collection of audio files
on a filesystem. The duration of the files varies across and within deployments
but is often much longer than the ideal length of individual training examples.
Annotations, which are the source of labeled data for training, are usually
stored as CSV, but the format of this also varies across datasets.

The *examplgen* Beam pipeline reads audio files and CSV label and metadata files
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

## Audio TFRecord inspector (dataset.print)

A debugging tool for inspecting the output of the examplegen pipeline.

Usage:

```
python3 -m multispecies_whale_detection.dataset.print --tfrecord_filepattern={output_directory}/tfrecords-*
```

This prints to the console a human-readable representation of the first few
records of an audio TFRecord dataset specified by the given file pattern. It is
mostly intended to be used by developers of this project, but it can be handy to
verify that a run of examplegen produced the expected form of output.
