# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
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

# Lint as: python3

import datasets
import os

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@article{marcus1999penn,
  title={Penn Treebank 3 LDC99T42},
  author={Marcus, Mitchell P and Santorini, Beatrice and Marcinkiewicz, Mary Ann and Taylor, Ann},
  journal={Web Download. Philadelphia: Linguistic Data Consortium},
  year={1999}
}
"""

class TbPosConfig(datasets.BuilderConfig):
    """BuilderConfig for Treebank POS"""

    def __init__(self, **kwargs):
        """BuilderConfig for Treebank POS.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TbPosConfig, self).__init__(**kwargs)


class TbPos(datasets.GeneratorBasedBuilder):
    """Treebank POS dataset."""

    BUILDER_CONFIGS = [
        TbPosConfig(name="TbPos", version=datasets.Version("1.0.0"), description="Treebank POS dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "labels": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                '"',
                                "''",
                                "#",
                                "$",
                                "(",
                                ")",
                                ",",
                                ".",
                                ":",
                                "``",
                                "CC",
                                "CD",
                                "DT",
                                "EX",
                                "FW",
                                "IN",
                                "JJ",
                                "JJR",
                                "JJS",
                                "-LRB-",
                                "LS",
                                "MD",
                                "NN",
                                "NNP",
                                "NNPS",
                                "NNS",
                                "PDT",
                                "POS",
                                "PRP",
                                "PRP$",
                                "RB",
                                "RBR",
                                "RBS",
                                "RP",
                                "-RRB-",
                                "SYM",
                                "TO",
                                "UH",
                                "VB",
                                "VBD",
                                "VBG",
                                "VBN",
                                "VBP",
                                "VBZ",
                                "WDT",
                                "WP",
                                "WP$",
                                "WRB",
                            ]
                        )
                    )
                }
            ),
            supervised_keys=None,
            homepage="https://catalog.ldc.upenn.edu/LDC99T42",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        paths = [os.path.join(os.environ.get('DATASET_DIR', './'), 'tb_pos', fn) for fn in ['train.tsv', 'dev.tsv', 'test.tsv']]
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": paths[0]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": paths[1]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": paths[2]})
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            pos_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "labels": pos_tags,
                        }
                        guid += 1
                        tokens = []
                        pos_tags = []
                else:
                    # tokens are tab separated
                    splits = line.rstrip().split("\t")
                    tokens.append(splits[0])
                    pos_tags.append(splits[1])
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "labels": pos_tags,
            }
