import logging
import os
from typing import Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from tqdm.auto import tqdm

import numpy as np

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
    def __init__(
        self,
        dataset: "Dataset",
        function: Callable[[dict, ], np.array],
        index_name: str,
        index_kwargs: Optional[dict] = None,
        size=768,
    ):
        self.dataset = dataset
        self.function = function
        self.index_name = index_name
        self.index_kwargs = index_kwargs if index_kwargs is not None else {}
        self.size = size

    def __getitem__(self, key: Union[int, slice, str]) -> Union[Dict, List]:
        return self.dataset[key]

    def query_index(self, query, k: int = 10) -> Tuple[List[float], List[int]]:
        raise NotImplementedError

    def init_index(self, **kwargs):
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
    def __init__(
        self,
        dataset: "Dataset",
        function: Callable[[dict, ], np.array],
        index_name: str,
        index_kwargs: Optional[dict] = None,
        size=768,
    ):
        super().__init__(dataset, function, index_name, index_kwargs, size)
        self.init_index(**index_kwargs or {})

    # indexes snippets with ElasticSearch
    def init_index(self, es_client, **kwargs):
        assert (
            _has_elasticsearch
        ), "You must install ElasticSearch to use SparseIndexedDataset. To do so you can run `pip install elasticsearch`"
        # Elasticsearch needs to be launched in another window, and a python client is declared with
        # > es_client = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
        self.es_client = es_client
        if self.function is not None:
            self._make_es_index()

    def _make_es_index(self):
        from tqdm.auto import tqdm

        es_client = self.es_client
        passages_dset = self.dataset
        index_name = self.index_name
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
        number_of_docs = passages_dset.num_rows
        progress = tqdm(unit="docs", total=number_of_docs)
        successes = 0

        def passage_generator():
            for passage in passages_dset:
                yield self.function(passage)

        # create the ES index
        for ok, action in es.helpers.streaming_bulk(client=es_client, index=index_name, actions=passage_generator(),):
            progress.update(1)
            successes += ok
        logger.info("Indexed %d documents" % (successes,))

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
    def __init__(
        self,
        dataset: "Dataset",
        function: Callable[[dict, ], np.array],
        index_name: str,
        index_kwargs: Optional[dict] = None,
        size=768,
    ):
        super().__init__(dataset, function, index_name, index_kwargs, size)
        self.array_file_path = os.path.join(HF_INDEXES_CACHE, index_name)
        self.init_index(**index_kwargs or {})

    def init_index(self, device=-1, **kwargs):
        assert (
            _has_faiss
        ), "You must install Faiss to use DenseIndexedDataset. To do so you can run `pip install faiss`"
        os.makedirs(HF_INDEXES_CACHE, exist_ok=True)
        if not os.path.exists(self.array_file_path):
            logger.info("Building dense index with name '{}'.".format(self.index_name))
            self._build_dense_index()
        else:
            logger.info("Dense index with name '{}' already exist, loading it.".format(self.index_name))
        self._load_dense_index(device)
        logger.info("Dense index '{}' loaded.".format(self.index_name))

    def _build_dense_index(self):
        assert self.function is not None
        fp = np.memmap(self.array_file_path, dtype="float32", mode="w+", shape=(self.dataset.num_rows, self.size))
        for i in tqdm(range(self.dataset.num_rows), total=self.dataset.num_rows):
            fp[i] = self.function(self.dataset[i])
        fp.flush()
        del fp

    def _load_dense_index(self, device=-1):
        """Load the numpy index into faiss. `device` is the index of the GPU, -1 for CPU"""
        fp = np.memmap(self.array_file_path, dtype="float32", mode="r", shape=(self.dataset.num_rows, self.size))
        index_flat = faiss.IndexFlatIP(self.size)
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
        assert queries.shape[1] == self.size
        scores, indices = self.faiss_index.search(queries, k)
        return scores[0], indices[0].astype(int)

    def query_index_batch(self, queries: np.array, k=10):
        assert len(queries.shape) == 2
        assert queries.shape[1] == self.size
        scores, indices = self.faiss_index.search(queries, k)
        return scores, indices.astype(int)

    def save(self, index_name: str):
        new_array_file_path = os.path.join(HF_INDEXES_CACHE, index_name)
        os.rename(self.array_file_path, new_array_file_path)
        self.index_name = index_name
        self.array_file_path = new_array_file_path
        logger.info("Dense index saved as {}".format(index_name))
