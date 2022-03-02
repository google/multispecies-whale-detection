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
import unittest

from multispecies_whale_detection import front_end
import tensorflow as tf
import tensorflow_probability as tfp


class TestFrontEnd(unittest.TestCase):

  def test_call_on_noise(self):
    sample_rate = 2000
    config = front_end.FrontEndConfig(sample_rate=sample_rate,)
    num_samples = sample_rate * 5
    noise_waveform = tf.random.normal([num_samples], 0, 1e-3)
    layer = front_end.FrontEnd(config)

    spectrogram = layer(noise_waveform)

    self.assertEqual([99, 128], spectrogram.shape)

  def test_call_on_noise_with_normalization(self):
    tf.random.set_seed(1413.0)  # to avoid flakiness

    sample_rate = 2000
    config = front_end.FrontEndConfig(
        sample_rate=sample_rate,
        per_channel_normalization=True,
        noise_floor_percentile=5.0,  # 5.0%
    )
    num_samples = sample_rate * 1
    noise_waveform = tf.random.normal([num_samples], 0, 1e-3)
    layer = front_end.FrontEnd(config)

    spectrogram = layer(noise_waveform)

    # The values are in a sensible dB range.
    self.assertTrue(tf.math.reduce_all(spectrogram >= -120.0).numpy())
    self.assertTrue(tf.math.reduce_all(spectrogram <= 120.0).numpy())

    # Running without normalization and shifting the noise floor percentile
    # across all channels should bring levels roughly into the same range as the
    # normalized spectrogram.
    unnormalized_spectrogram = front_end.FrontEnd(
        dataclasses.replace(config, per_channel_normalization=False))(
            noise_waveform)
    floor = tfp.stats.percentile(unnormalized_spectrogram,
                                 config.noise_floor_percentile)
    margin_db = 12.0
    self.assertLess(
        tf.math.reduce_max(
            tf.abs((unnormalized_spectrogram - floor) - spectrogram)),
        margin_db)
    # Note that the normalization has made a large adjustment in overall level.
    self.assertGreater(
        tf.math.reduce_max(
            tf.abs(unnormalized_spectrogram - spectrogram)),
        margin_db + 120.0)


if __name__ == '__main__':
  unittest.main()
