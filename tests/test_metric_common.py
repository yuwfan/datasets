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
import tempfile
import os
import json

from absl.testing import parameterized

from nlp import DownloadConfig, hf_api, load_metric, GenerateMode, cached_path, logging

from .utils import aws, local, slow

logger = logging.get_logger(__name__)

def get_aws_metric_names():
    api = hf_api.HfApi()
    # fetch all metric names
    metrics = [x.id for x in api.metric_list()]
    return [{"testcase_name": x, "metric_name": x} for x in metrics]


def get_local_metric_names():
    metrics = [metric_dir.split("/")[-2] for metric_dir in glob.glob("./metrics/*/")]
    return [{"testcase_name": x, "metric_name": x} for x in metrics]


@parameterized.named_parameters(get_aws_metric_names())
@aws
class AWSMetricTest(parameterized.TestCase):
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
                metric_name, config_name=config_name, data_dir=temp_data_dir, download_config=download_config, download_mode=GenerateMode.FORCE_REDOWNLOAD,
            )

            parameters = inspect.signature(metric._compute).parameters
            self.assertTrue("predictions" in parameters)
            self.assertTrue("references" in parameters)
            self.assertTrue(all([p.kind != p.VAR_KEYWORD for p in parameters.values()]))  # no **kwargs

    @slow
    def test_predictions_real_metric(self, metric_name):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = os.path.join("./metrics", metric_name, "test.JSON")
            try:
                test_json_path = cached_path(base_path, cache_dir=tmp_dir)
            except FileNotFoundError:
                logger.info(f"No test file for metric {metric_name}, skipping.")
            else:
                logger.info(f"Test file found for metric {metric_name}, checking predictions.")
                with open(test_json_path, encoding="utf-8") as f:
                    test_inputs = json.load(f)

                score = test_inputs.pop("__score")

                download_config = DownloadConfig()
                download_config.force_download = True
                config_name = None
                if metric_name == "glue":
                    config_name = "sst2"
                metric = load_metric(
                    metric_name, config_name=config_name, data_dir=tmp_dir, download_config=download_config, download_mode=GenerateMode.FORCE_REDOWNLOAD,
                )

                pred_score = metric.compute(**test_inputs)
                self.assertDictEqual(score, pred_score)
