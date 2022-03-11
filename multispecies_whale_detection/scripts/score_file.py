""" Utility to score a SavedModel on all FLAC files in a given directory."""

import os

from typing import Sequence

from absl import app
from absl import flags
import resampy
import soundfile
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('saved_model', None, 'model to load.')
flags.DEFINE_string('audio_dir', None, 'directory of files to analyze.')
flags.DEFINE_integer('sample_rate', None, 'sample rate expected by the model.')
flags.DEFINE_float('window_duration', 1.0, 'duration of model input.')


def main(argv: Sequence[str]) -> None:
  del argv
  model = tf.keras.models.load_model(FLAGS.saved_model)
  for filename in os.listdir(FLAGS.audio_dir):
    print(filename)
    if not filename.endswith('.flac'):
      continue
    full_filename = os.path.join(FLAGS.audio_dir, filename)
    data, original_sample_rate = soundfile.read(full_filename)
    resampled_audio = resampy.resample(data, original_sample_rate,
                                       FLAGS.sample_rate)
    window_duration_samples = int(FLAGS.window_duration * FLAGS.sample_rate)
    model_input = tf.signal.frame(
        signal=resampled_audio,
        frame_length=window_duration_samples,
        frame_step=window_duration_samples,
    )

    scores = model(model_input)
    print(scores)


if __name__ == '__main__':
  flags.mark_flag_as_required('saved_model')
  flags.mark_flag_as_required('audio_dir')
  flags.mark_flag_as_required('sample_rate')

  app.run(main)
