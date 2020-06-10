import logging
from typing import List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm


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


class MissingIndex(Exception):
    pass


class IndexableMixin:
    """Add indexing features to classes"""

    def __init__(self):
        self._index: Optional[BaseIndex] = None

    def __getitem__(self, key):
        raise NotImplementedError

    def _indexable_examples(self):
        raise NotImplementedError

    def _indexable_embeddings(self):
        raise NotImplementedError

    def _check_index_is_initialized(self):
        if self._index is None:
            raise MissingIndex("Index not initialized yet. Please make sure that you call `init_index` first.")

    def init_index(self, index_type="dense", **index_kwargs):
        assert index_type in ["dense", "sparse"]
        if index_type == "sparse":
            self._index = SparseIndex(**index_kwargs)
            self._index.add_passages(self._indexable_examples())
        else:
            self._index = DenseIndex(**index_kwargs)
            self._index.add_embeddings(self._indexable_embeddings())

    def query_index(self, query, k: int = 10) -> Tuple[List[float], List[int]]:
        self._check_index_is_initialized()
        return self._index.query_index(query, k)

    def query_index_batch(self, queries, k: int = 10) -> Tuple[List[List[float]], List[List[int]]]:
        self._check_index_is_initialized()
        return self._index.query_index_batch(queries, k)

    def get_nearest(self, query, k: int = 10) -> Tuple[List[float], List[dict]]:
        self._check_index_is_initialized()
        scores, indices = self.query_index(query, k)
        return scores, [self[int(i)] for i in indices]

    def get_nearest_batch(self, queries, k: int = 10) -> Tuple[List[List[float]], List[List[dict]]]:
        self._check_index_is_initialized()
        total_scores, total_indices = self.query_index_batch(queries, k)
        return total_scores, [[self[int(i)] for i in indices] for indices in total_indices]


class BaseIndex:
    def query_index(self, query, k: int = 10) -> Tuple[List[float], List[int]]:
        raise NotImplementedError

    def query_index_batch(self, queries, k: int = 10) -> Tuple[List[List[float]], List[List[int]]]:
        total_scores, total_indices = [], []
        for query in queries:
            scores, indices = self.query_index(query, k)
            total_scores.append(scores)
            total_indices.append(indices)
        return total_scores, total_indices


class SparseIndex(BaseIndex):
    def __init__(self, es_client, index_name: str, column: Optional[str] = None):
        # Elasticsearch needs to be launched in another window, and a python client is declared with
        # > es_client = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
        self.es_client = es_client
        self.index_name = index_name
        self.column = column
        assert (
            _has_elasticsearch
        ), "You must install ElasticSearch to use SparseIndexedDataset. To do so you can run `pip install elasticsearch`"

    def add_passages(self, examples):
        # TODO: don't rebuild if it already exists
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
        self.es_client.indices.create(index=index_name, body=index_config)
        number_of_docs = len(examples)
        progress = tqdm(unit="docs", total=number_of_docs)
        successes = 0

        def passage_generator():
            if self.column is not None:
                for example in examples:
                    yield example[self.column]
            else:
                for example in examples:
                    yield example

        # create the ES index
        for ok, action in es.helpers.streaming_bulk(
            client=self.es_client, index=index_name, actions=passage_generator(),
        ):
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


class DenseIndex(BaseIndex):
    def __init__(self, device: int = -1):
        self.device = device
        self.faiss_index = None
        assert (
            _has_faiss
        ), "You must install Faiss to use SparseIndexedDataset. To do so you can run `pip install faiss-cpu`"

    def add_embeddings(self, embeddings: np.array):
        index_flat = faiss.IndexFlatIP(embeddings.shape[1])
        if self.device > -1:
            faiss_res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(faiss_res, self.device, index_flat)
        else:
            self.faiss_index = index_flat
        self.faiss_index.add(embeddings)

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
