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
"""Ops for signal processing that prepares neural network input."""
import dataclasses
import enum
from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp


def amplitude_ratio_to_db(x):
  return 20.0 * tf.math.log(x / 10.0)


def db_to_amplitude_ratio(x):
  return tf.math.pow(10.0, x / 20.0)


@dataclasses.dataclass(frozen=True)
class NoiseFloorConfig:
  """Parameters for subtracting estimated noise floors from each STFT row.

  This equalizes overall levels and frequency responses and is useful for
  datasets where these are inconsistent.
  """

  percentile: float = 5.0
  """Percentile of values in a frequency bin to deem the noise floor."""
  smoother_extent_hz: Optional[float] = 20.0
  """Scale of average pooling over frequency done before taking percentile.

  Narrowband signals lasting for a large fraction of the context window can have
  high amplitude at a percentile that is suitable for other bands, resulting in
  too much attenuation in the band with the signal. Averaging over nearby bands
  before taking the percentile can mitigate this effect.
  """


@dataclasses.dataclass(frozen=True)
class SpectrogramConfig:
  """Configuration data object for the Spectrogram Keras layer.

  The default values for attributes related to time scale differ significantly
  from those conventionally used for speech. It is recommended that the caller
  verify whether they are appropriate for each particular use case.
  """

  sample_rate: int = 2000
  """Sample rate of the input waveform."""
  frame_seconds: float = 0.1
  """Duration of STFT frames."""
  hop_seconds: float = 0.05
  """Interval between successive STFT frames."""
  log_stabilizer: float = 1e-6
  """Used in log((STFT magnitude) + stabilizer) to avoid division by zero."""
  lower_bound_hz: float = 8.0
  """Frequency below which the spectrogram should be truncated.

  This is intended as a way of ignoring low-frequency noise.
  """

  normalization: Optional[NoiseFloorConfig] = None
  """Type of and parameters for the normalization strategy."""


def spectrogram(
    waveform: tf.Tensor,
    sample_rate: tf.Tensor,
    frame_seconds: float = 0.025,
    hop_seconds: float = 0.01,
) -> tf.Tensor:
  """Computes a linear-frequency-scale magnitude spectrogram.

  This is a thin convenience wrapper of tf.signal.stft. The additions are a) the
  conversion of STFT frame duration and hop from seconds to samples and b)
  taking the magnitude of the complex tensor returned by tf.signal.stft.

  Args:
    waveform: A [..., samples] float32/float64 Tensor of real-valued signals.
    sample_rate: The number of samples per second.
    frame_seconds: The duration of a single STFT frame.
    hop_seconds: The interval between the starts of consecutive STFT frames.

  Returns:
    Spectrogram of waveform where the frequency scale is linear and the elements
    are real magnitudes.
  """
  frame_length = int(frame_seconds * sample_rate)
  frame_step = int(hop_seconds * sample_rate)
  tf.assert_greater(frame_length, 0)
  tf.assert_greater(frame_step, 0)
  stft = tf.signal.stft(waveform,
                        frame_length=frame_length,
                        frame_step=frame_step,
                        pad_end=False)
  return tf.abs(stft)


def _subtract_noise_floor(sgram: tf.Tensor, config: NoiseFloorConfig,
                          hz_per_bin: float) -> tf.Tensor:
  """Subtracts an estimated noise floor from a spectrogram.

  See :py:class:`NoiseFloorConfig` for how the estimate is made.

  Args:
    sgram: Batch of spectrograms of shape [N, T, F] or single spectrogram of
      shape [T, F].
    config: Parameters for estimating the noise floor.
    hz_per_bin: Frequency resolution of sgram.

  Returns:
    Spectrogram of the same shape as the input with an estimated noise level
    subtracted from each frequency band.
  """
  if config.smoother_extent_hz:
    smoother_len = int(config.smoother_extent_hz / hz_per_bin)
    is_batch = sgram.shape.rank > 2
    if is_batch:
      smoother_input = sgram
    else:
      smoother_input = tf.expand_dims(sgram, 0)
    smoothed = tf.squeeze(
        tf.nn.avg_pool2d(tf.expand_dims(smoother_input, -1), [1, smoother_len],
                         1, 'SAME'), -1)
    if not is_batch:
      smoothed = tf.squeeze(smoothed, 0)
    floor_input = smoothed
  else:
    floor_input = sgram
  floor = tfp.stats.percentile(floor_input,
                               config.percentile,
                               axis=1,
                               keepdims=True)
  return sgram - floor


class Spectrogram(tf.keras.layers.Layer):
  """Converts audio from waveform to spectrogram.

  This layer simplifies building a Keras model that takes a raw waveform as
  input and passes it to a CNN. It is initialized with a configuration data
  object, SpectrogramConfig, which collects commonly used fixed parameters.

  The frequency scale is always linear. (A future version will add mel frequency
  scaling.)

  It optionally normalizes by subtracting a per-channel (frequency bin) noise
  floor, which is estimated as a percentile of the STFT magnitudes in that
  channel.
  """

  def __init__(self, config=None):
    """Initializes this layer.

    Args:
      config: Data structure of parameters that can be used to tune the
        spectrogram to different time scales, distances to source, level
        variability across endpoints, etc.
    """
    super(Spectrogram, self).__init__()
    if not config:
      config = SpectrogramConfig()
    self._config = config

  def call(self, waveform, training=False):
    sample_rate = self._config.sample_rate
    magnitude = spectrogram(waveform,
                            sample_rate=sample_rate,
                            frame_seconds=self._config.frame_seconds,
                            hop_seconds=self._config.hop_seconds)
    num_frequency_bins = magnitude.shape[-1]
    hz_per_bin = sample_rate / 2 / num_frequency_bins

    db = amplitude_ratio_to_db(magnitude + self._config.log_stabilizer)

    normalization = self._config.normalization
    if not normalization:
      sgram = db
    elif isinstance(normalization, NoiseFloorConfig):
      sgram = _subtract_noise_floor(db, normalization, hz_per_bin)
    else:
      raise TypeError(f'unknown normalization type {type(normalization)}')

    sgram = sgram[..., int(self._config.lower_bound_hz / hz_per_bin):]

    return sgram

  def get_config(self):
    """Implements Layer.get_config."""
    config = dataclasses.asdict(self._config)
    normalization = self._config.normalization
    if normalization:
      config.update(normalization_type=normalization.__class__.__name__)
    return config

  @classmethod
  def from_config(cls, config):
    """Implements Layer.from_config."""
    normalization_config = config.pop('normalization', None)
    if normalization_config:
      normalization_class = globals()[config.pop('normalization_type')]
      normalization = normalization_class(**normalization_config)
      del normalization_class
    else:
      normalization = None
    return cls(SpectrogramConfig(normalization=normalization, **config))
