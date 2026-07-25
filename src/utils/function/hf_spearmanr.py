# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""Spearman correlation coefficient metric."""

from scipy.stats import spearmanr

import datasets

_DESCRIPTION = """
The Spearman rank-order correlation coefficient is a measure of the monotonic relationship between two datasets.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Predicted values, as returned by a model.
    references: Ground truth values.
Returns:
    spearmanr: Spearman correlation coefficient.
"""

_CITATION = """\
@article{2020SciPy-NMeth, author={Virtanen, Pauli and others}, title={{SciPy} 1.0: Fundamental Algorithms for
  Scientific Computing in Python}, journal={Nature Methods}, year={2020}, volume={17}, pages={261--272}}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Spearmanr(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float"),
                    "references": datasets.Value("float"),
                }
            ),
            reference_urls=["https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html"],
        )

    def _compute(self, predictions, references):
        return {"spearmanr": float(spearmanr(references, predictions)[0])}
