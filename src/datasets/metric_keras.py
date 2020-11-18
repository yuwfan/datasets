from typing import List, Type

import tensorflow as tf
from tensorflow import keras

from datasets.metric import Metric


class SharedMetricOutput:
    def __init__(self):
        self._output = None

    def is_set(self):
        return self._output is not None

    def set(self, output: dict):
        self._output = output

    def get(self, output_value):
        return self._output[output_value]

    def reset(self):
        self._output = None


def get_metric_full_name(metric: Metric, return_value: str) -> str:
    return f"{metric.name}.{metric.config_name}.{return_value}"


class KerasMetric(keras.metrics.Metric):
    def __init__(
        self, metric: Metric, return_value: str, shared_output: SharedMetricOutput, update_state_enabled=True
    ):
        super().__init__(name=get_metric_full_name(metric, return_value), dtype=tf.float32)
        self.metric = metric
        self.return_value = return_value
        self.shared_output = shared_output
        self.update_state_enabled = update_state_enabled

    def update_state(self, y_true, y_pred):
        if self.update_state_enabled:
            self.metric.add_batch(references=y_true, predictions=y_pred)

    def result(self):
        if not self.shared_output.is_set():
            self.shared_output.set(self.metric.compute())
        return self.shared_output.get(self.return_value)

    def reset_states(self):
        self.metric.reset()
        self.shared_output.reset()


def convert_metric_to_keras(metric: Metric) -> List[keras.metrics.Metric]:
    if not metric.output_names:
        raise ValueError(f"Unable to load keras metric since metric.output_names for metric {metric} is empty.")
    keras_metrics = []
    shared_output = SharedMetricOutput()
    for i, output_name in enumerate(metric.output_names):
        update_state_enabled = i == 0
        keras_metrics.append(
            KerasMetric(
                metric=metric,
                return_value=output_name,
                shared_output=shared_output,
                update_state_enabled=update_state_enabled,
            )
        )
    return keras_metrics
