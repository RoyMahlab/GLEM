# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""Recall metric."""

from sklearn.metrics import recall_score

import datasets

_DESCRIPTION = """
Recall is the fraction of the positive examples that were correctly labeled by the model as positive. It is
computed as: Recall = TP / (TP + FN)
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Predicted labels, as returned by a model.
    references: Ground truth labels.
    labels: The set of labels to include when average is not set to 'binary'.
    pos_label: The class to report if average is 'binary' and the data is binary.
    average: One of 'micro', 'macro', 'samples', 'weighted', 'binary' or None.
    sample_weight: Sample weights.
Returns:
    recall: Recall score.
"""

_CITATION = """\
@article{scikit-learn, title={Scikit-learn: Machine Learning in {P}ython}, author={Pedregosa et al.},
  journal={Journal of Machine Learning Research}, volume={12}, pages={2825--2830}, year={2011}}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Recall(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html"],
        )

    def _compute(self, predictions, references, labels=None, pos_label=1, average="binary", sample_weight=None):
        score = float(
            recall_score(
                references,
                predictions,
                labels=labels,
                pos_label=pos_label,
                average=average,
                sample_weight=sample_weight,
                zero_division=0,
            )
        )
        return {"recall": score}
