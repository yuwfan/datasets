# coding=utf-8
# Copyright 2020 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import inspect
import os
import tempfile

from absl.testing import parameterized

from datasets import DownloadConfig, hf_api, load_metric

from .utils import local, remote, require_tf, slow


def get_local_metric_names():
    metrics = [metric_dir.split(os.sep)[-2] for metric_dir in glob.glob("./metrics/*/")]
    return [{"testcase_name": x, "metric_name": x} for x in metrics]


def get_remote_metric_names():
    api = hf_api.HfApi()
    # fetch all metric names
    metrics = [x.id for x in api.metric_list()]
    return [{"testcase_name": x, "metric_name": x} for x in metrics]


@parameterized.named_parameters(get_remote_metric_names())
@remote
class RemoteMetricTest(parameterized.TestCase):
    metric_name = None

    @slow
    def test_load_real_metric(self, metric_name):
        with tempfile.TemporaryDirectory() as temp_data_dir:
            download_config = DownloadConfig()
            download_config.force_download = True
            config_name = None
            if metric_name == "glue":
                config_name = "sst2"
            metric = load_metric(
                metric_name, config_name=config_name, data_dir=temp_data_dir, download_config=download_config
            )

            parameters = inspect.signature(metric._compute).parameters
            self.assertTrue("predictions" in parameters)
            self.assertTrue("references" in parameters)
            self.assertTrue(all([p.kind != p.VAR_KEYWORD for p in parameters.values()]))  # no **kwargs
            del metric

    @slow
    @require_tf
    def test_load_real_keras_metric(self, metric_name):
        from datasets import convert_metric_to_keras

        with tempfile.TemporaryDirectory() as temp_data_dir:
            download_config = DownloadConfig()
            download_config.force_download = True
            config_name = None
            if metric_name == "glue":
                config_name = "sst2"
            metric = load_metric(
                metric_name, config_name=config_name, data_dir=temp_data_dir, download_config=download_config
            )
            keras_metrics = convert_metric_to_keras(metric)
            self.assertTrue(len(keras_metrics) > 0)
            del keras_metrics, metric


@parameterized.named_parameters(get_local_metric_names())
@local
class LocalMetricTest(parameterized.TestCase):
    metric_name = None

    @slow
    def test_load_real_metric(self, metric_name):
        with tempfile.TemporaryDirectory() as temp_data_dir:
            download_config = DownloadConfig()
            download_config.force_download = True
            config_name = None
            if metric_name == "glue":
                config_name = "sst2"
            metric = load_metric(
                metric_name, config_name=config_name, data_dir=temp_data_dir, download_config=download_config
            )

            parameters = inspect.signature(metric._compute).parameters
            self.assertTrue("predictions" in parameters)
            self.assertTrue("references" in parameters)
            self.assertTrue(all([p.kind != p.VAR_KEYWORD for p in parameters.values()]))  # no **kwargs
            del metric

    @slow
    @require_tf
    def test_load_real_keras_metric(self, metric_name):
        from datasets import convert_metric_to_keras

        with tempfile.TemporaryDirectory() as temp_data_dir:
            download_config = DownloadConfig()
            download_config.force_download = True
            config_name = None
            if metric_name == "glue":
                config_name = "sst2"
            metric = load_metric(
                metric_name, config_name=config_name, data_dir=temp_data_dir, download_config=download_config
            )
            keras_metrics = convert_metric_to_keras(metric)
            self.assertTrue(len(keras_metrics) > 0)
            del keras_metrics, metric
