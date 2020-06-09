import logging
import os
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm.auto import tqdm

from .utils.file_utils import HF_INDEXES_CACHE


if TYPE_CHECKING:
    from .arrow_dataset import Dataset

try:
    import elasticsearch as es

    _has_elasticsearch = True
except ImportError:
    _has_elasticsearch = False

try:
    import faiss

    _has_faiss = True
except ImportError:
    _has_faiss = False


logger = logging.getLogger(__name__)


class BaseEncodedDataset:
    def __init__(self, dataset: "Dataset"):
        self.dataset = dataset

    def __getitem__(self, key: Union[int, slice, str]) -> Union[Dict, List]:
        return self.dataset[key]

    def query_index(self, query, k: int = 10) -> Tuple[List[float], List[int]]:
        raise NotImplementedError

    def query_index_batch(self, queries, k: int = 10) -> Tuple[List[List[float]], List[List[int]]]:
        total_scores, total_indices = [], []
        for query in queries:
            scores, indices = self.query_index(query, k)
            total_scores.append(scores)
            total_indices.append(indices)
        return total_scores, total_indices

    def get_nearest(self, query, k: int = 10) -> Tuple[List[float], List[dict]]:
        scores, indices = self.query_index(query, k)
        return scores, [self.dataset[int(i)] for i in indices]

    def get_nearest_batch(self, queries, k: int = 10) -> Tuple[List[List[float]], List[List[dict]]]:
        total_scores, total_indices = self.query_index_batch(queries, k)
        return total_scores, [[self.dataset[int(i)] for i in indices] for indices in total_indices]


class SparseEncodedDataset(BaseEncodedDataset):
    def __init__(self, dataset: "Dataset", es_client, index_name: str):
        super().__init__(dataset)
        # Elasticsearch needs to be launched in another window, and a python client is declared with
        # > es_client = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
        self.es_client = es_client
        self.index_name = index_name
        assert (
            _has_elasticsearch
        ), "You must install ElasticSearch to use SparseIndexedDataset. To do so you can run `pip install elasticsearch`"

    @staticmethod
    def build_from_dataset(
        dataset: "Dataset", function: Callable, es_client, index_name: Optional[str]
    ) -> "SparseEncodedDataset":
        # TODO: don't rebuild if it already exists
        index_name = index_name or os.path.basename(NamedTemporaryFile().name)
        index_config = {
            "settings": {
                "number_of_shards": 1,
                "analysis": {"analyzer": {"stop_standard": {"type": "standard", " stopwords": "_english_"}}},
            },
            "mappings": {
                "properties": {
                    "article_title": {"type": "text", "analyzer": "standard", "similarity": "BM25"},
                    "section_title": {"type": "text", "analyzer": "standard", "similarity": "BM25"},
                    "passage_text": {"type": "text", "analyzer": "standard", "similarity": "BM25"},
                }
            },
        }
        es_client.indices.create(index=index_name, body=index_config)
        number_of_docs = dataset.num_rows
        progress = tqdm(unit="docs", total=number_of_docs)
        successes = 0

        def passage_generator():
            for passage in dataset:
                yield function(passage)

        # create the ES index
        for ok, action in es.helpers.streaming_bulk(client=es_client, index=index_name, actions=passage_generator(),):
            progress.update(1)
            successes += ok
        logger.info("Indexed %d documents" % (successes,))
        return SparseEncodedDataset(dataset, es_client, index_name)

    def query_index(self, query, k=10):
        response = self.es_client.search(
            index=self.index_name,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["article_title", "section_title", "passage_text^2"],
                        "type": "cross_fields",
                    }
                },
                "size": k,
            },
        )
        hits = response["hits"]["hits"]
        return [hit["_score"] for hit in hits], [hit["_id"] for hit in hits]


class DenseEncodedDataset(BaseEncodedDataset):
    def __init__(self, dataset: "Dataset", encodings_filename: Optional[str] = None, device: int = -1):
        super().__init__(dataset)
        self.encodings_filename = encodings_filename or os.path.basename(NamedTemporaryFile().name)
        self.array_file_path = os.path.join(HF_INDEXES_CACHE, self.encodings_filename)
        assert (
            _has_faiss
        ), "You must install Faiss to use DenseIndexedDataset. To do so you can run `pip install faiss`"
        self._load_dense_index(device)
        logger.info("Dense index '{}' loaded.".format(self.encodings_filename))

    @staticmethod
    def build_from_dataset(
        dataset: "Dataset", function: Callable, encodings_filename: Optional[str] = None, **kwargs
    ) -> "DenseEncodedDataset":
        encodings_filename = encodings_filename or os.path.basename(NamedTemporaryFile().name)
        array_file_path = os.path.join(HF_INDEXES_CACHE, encodings_filename)
        os.makedirs(HF_INDEXES_CACHE, exist_ok=True)
        if os.path.exists(array_file_path):
            logger.info("Dense index '{}' already exist, loading it.".format(encodings_filename))
            return DenseEncodedDataset(dataset, encodings_filename=encodings_filename, **kwargs)
        logger.info("Building dense index '{}'.".format(encodings_filename))
        size = len(function(dataset[0]).flatten())
        fp = np.memmap(array_file_path, dtype="float32", mode="w+", shape=(dataset.num_rows, size))
        for i in tqdm(range(dataset.num_rows), total=dataset.num_rows):
            fp[i] = function(dataset[i])
        fp.flush()
        del fp
        return DenseEncodedDataset(dataset, encodings_filename=encodings_filename, **kwargs)

    def _load_dense_index(self, device=-1):
        """Load the numpy index into faiss. `device` is the index of the GPU, -1 for CPU"""
        fp = np.memmap(self.array_file_path, dtype="float32", mode="r").reshape(self.dataset.num_rows, -1)
        index_flat = faiss.IndexFlatIP(fp.shape[1])
        if device > -1:
            faiss_res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(faiss_res, device, index_flat)
        else:
            self.faiss_index = index_flat
        self.faiss_index.add(fp)
        fp.flush()

    def query_index(self, query: np.array, k=10):
        assert len(query.shape) < 3
        queries = query.reshape(1, -1)
        scores, indices = self.faiss_index.search(queries, k)
        return scores[0], indices[0].astype(int)

    def query_index_batch(self, queries: np.array, k=10):
        assert len(queries.shape) == 2
        assert queries.shape[1] == self.size
        scores, indices = self.faiss_index.search(queries, k)
        return scores, indices.astype(int)

    def save(self, encodings_filename: str):
        new_array_file_path = os.path.join(HF_INDEXES_CACHE, encodings_filename)
        os.rename(self.array_file_path, new_array_file_path)
        self.encodings_filename = encodings_filename
        self.array_file_path = new_array_file_path
        logger.info("Dense index saved as {}".format(encodings_filename))
