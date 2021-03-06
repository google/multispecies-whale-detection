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

import dataclasses
import json
import unittest

from multispecies_whale_detection import front_end
import tensorflow as tf
import tensorflow_probability as tfp


class TestFrontEnd(unittest.TestCase):

  def test_amplitude_db_scaling(self):
    ratios = tf.constant([0.1, 0.5, 1.0, 2.0, 5.0], tf.float32)
    db = front_end.amplitude_ratio_to_db(ratios)
    db_ten_x = front_end.amplitude_ratio_to_db(10.0 * ratios)
    self.assertTrue(tf.reduce_max(tf.math.abs((db + 20.0) - db_ten_x)) < 1e-6)

  def test_power_db_scaling(self):
    ratios = tf.constant([0.1, 0.5, 1.0, 2.0, 5.0], tf.float32)
    db = front_end.power_ratio_to_db(ratios)
    db_ten_x = front_end.power_ratio_to_db(10.0 * ratios)
    self.assertTrue(tf.reduce_max(tf.math.abs((db + 10.0) - db_ten_x)) < 1e-6)

  def test_amplitude_to_db_round_trip(self):
    ratios = tf.constant([0.1, 0.5, 1.0, 2.0, 5.0], tf.float32)
    db = front_end.amplitude_ratio_to_db(ratios)
    ratios_from_db = front_end.db_to_amplitude_ratio(db)
    self.assertTrue(tf.reduce_max(tf.math.abs(ratios - ratios_from_db)) < 1e-6)

  def test_power_and_amplitude_round_trip(self):
    ratios = tf.constant([0.1, 0.5, 1.0, 2.0, 5.0], tf.float32)
    power_ratios = tf.math.square(ratios)
    db = front_end.power_ratio_to_db(power_ratios)
    ratios_from_db = front_end.db_to_amplitude_ratio(db)
    self.assertTrue(tf.reduce_max(tf.math.abs(ratios - ratios_from_db)) < 1e-6)

  def test_spectrogram_layer(self):
    tf.random.set_seed(3141)  # to avoid flakiness

    sample_rate = 2000
    config = front_end.SpectrogramConfig(sample_rate=sample_rate,)
    num_samples = sample_rate * 5
    noise_waveform = tf.random.normal([num_samples], 0, 1e-3)
    layer = front_end.Spectrogram(config)

    spectrogram = layer(noise_waveform)

    self.assertEqual([99, 129], spectrogram.shape)

  def test_spectrogram_layer_with_mel_scaling(self):
    tf.random.set_seed(1413)  # to avoid flakiness

    sample_rate = 2000
    frequency_scaling = front_end.MelScalingConfig(
        num_mel_bins=80,
        lower_edge_hz=20.0,
    )
    config = front_end.SpectrogramConfig(
        sample_rate=sample_rate,
        frequency_scaling=frequency_scaling,
    )
    num_samples = sample_rate * 1
    noise_waveform = tf.random.normal([num_samples], 0, 1e-3)
    layer = front_end.Spectrogram(config)

    spectrogram = layer(noise_waveform)

    self.assertEqual(frequency_scaling.num_mel_bins, spectrogram.shape[-1])

  def test_mel_scaling_upper_edge_exception(self):
    tf.random.set_seed(1413)  # to avoid flakiness

    sample_rate = 2000
    frequency_scaling = front_end.MelScalingConfig(
        num_mel_bins=80,
        lower_edge_hz=20.0,
        upper_edge_hz=(sample_rate / 2 * 1.1),  # > Nyquist
    )
    config = front_end.SpectrogramConfig(
        sample_rate=sample_rate,
        frequency_scaling=frequency_scaling,
    )
    num_samples = sample_rate * 1
    noise_waveform = tf.random.normal([num_samples], 0, 1e-3)
    layer = front_end.Spectrogram(config)

    with self.assertRaises(ValueError):
      _ = layer(noise_waveform)

  def test_spectrogram_layer_with_cropping(self):
    tf.random.set_seed(1413)  # to avoid flakiness

    sample_rate = 2000
    num_samples = sample_rate * 1
    noise_waveform = tf.random.normal([num_samples], 0, 1e-3)
    num_bins = 128
    hz_per_bin = sample_rate / 2 / num_bins

    for lower_edge_hz in [0.0, 50.0, 150.0, 500.0]:
      frequency_scaling = front_end.CropFrequencyConfig(
          lower_edge_hz=lower_edge_hz)
      config = front_end.SpectrogramConfig(
          sample_rate=sample_rate,
          frequency_scaling=frequency_scaling,
      )
      layer = front_end.Spectrogram(config)

      spectrogram = layer(noise_waveform)

      self.assertEqual(num_bins - int(lower_edge_hz / hz_per_bin) + 1,
                       spectrogram.shape[-1])

  def test_spectrogram_layer_with_normalization(self):
    tf.random.set_seed(1413)  # to avoid flakiness

    sample_rate = 2000
    normalization = front_end.NoiseFloorConfig(percentile=5.0)
    config = front_end.SpectrogramConfig(
        sample_rate=sample_rate,
        normalization=normalization,
    )
    num_samples = sample_rate * 1
    noise_waveform = tf.random.normal([num_samples], 0,
                                      front_end.db_to_amplitude_ratio(-30.0))
    layer = front_end.Spectrogram(config)

    spectrogram = layer(noise_waveform)

    # The values are in a sensible dB range.
    self.assertTrue(tf.math.reduce_all(spectrogram >= -40.0).numpy())
    self.assertTrue(tf.math.reduce_all(spectrogram <= 30.0).numpy())

    # Running without normalization and shifting the noise floor percentile
    # across all channels should bring levels roughly into the same range as the
    # normalized spectrogram.
    unnormalized_spectrogram = front_end.Spectrogram(
        dataclasses.replace(config, normalization=None))(
            noise_waveform)
    floor = tfp.stats.percentile(unnormalized_spectrogram,
                                 normalization.percentile)
    margin_db = 6.0
    self.assertLess(
        tf.math.reduce_max(
            tf.abs((unnormalized_spectrogram - floor) - spectrogram)),
        margin_db)
    # Note that the normalization has made a large adjustment in overall level.
    self.assertGreater(
        tf.math.reduce_max(tf.abs(unnormalized_spectrogram - spectrogram)),
        24.0)

  def test_spectrogram_config_serialization(self):
    default_config = front_end.SpectrogramConfig()
    non_default_config = front_end.SpectrogramConfig(
        frame_seconds=0.025,
        hop_seconds=0.0125,
        frequency_scaling=front_end.MelScalingConfig(
            num_mel_bins=64,
            lower_edge_hz=20.0,
            upper_edge_hz=900.0,
        ),
        normalization=front_end.NoiseFloorConfig(
            percentile=0.3,
            smoother_extent_hz=None,
        ),
    )
    assert default_config != non_default_config
    layer = front_end.Spectrogram(non_default_config)
    default_layer = front_end.Spectrogram(default_config)

    keras_config = layer.get_config()
    reloaded_layer = front_end.Spectrogram.from_config(keras_config)

    waveform = tf.random.normal([4000], 0, 1e-3)
    non_default_output = layer(waveform)
    default_output = default_layer(waveform)
    reloaded_output = reloaded_layer(waveform)
    tolerance = 1e-4
    # Change from default settings made the output not close to equal, what's
    # more, made it not the same shape.
    assert non_default_output.shape != default_output.shape
    # But the output of the reloaded layer matches the output from before
    # saving.
    self.assertLess(
        tf.math.reduce_max(non_default_output - reloaded_output), tolerance)


class TestSpectrogram(unittest.TestCase):

  def test_bad_config(self):
    with self.assertRaises(AssertionError):
      # Wrong: NoiseFloorConfig as FrequencyScalingConfig
      config = front_end.SpectrogramConfig(frequency_scaling=front_end.NoiseFloorConfig())
      front_end.Spectrogram(config)
    with self.assertRaises(AssertionError):
      # Wrong: FrequencyScalingConfig as NoiseFloorConfig
      config = front_end.SpectrogramConfig(normalization=front_end.MelScalingConfig())
      front_end.Spectrogram(config)

    # Make sure default config is correct.
    front_end.Spectrogram(front_end.SpectrogramConfig())


if __name__ == '__main__':
  unittest.main()
