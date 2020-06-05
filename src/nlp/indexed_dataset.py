import logging
import math
import os
from time import time
from typing import Dict, List, Tuple, Union, Callable, Optional

import numpy as np

from .arrow_dataset import Dataset
from .utils.file_utils import HF_INDEXES_CACHE


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


class BaseEmbedModel:

    def embed_documents(self, texts: List[str], titles: Optional[List[str]] = None, tokenizer: Optional[Callable[[str], List[str]]] = None) -> np.array:
        raise NotImplementedError

    def embed_queries(self, queries: List[str], tokenizer: Optional[Callable[[str], List[str]]] = None) -> np.array:
        raise NotImplementedError


class BaseIndexedDataset:
    def __init__(
        self,
        dataset: Dataset,
        index_name: str,
        embed_model: BaseEmbedModel,
        index_kwargs: Optional[dict] = None,
        embed_tokenizer: Optional[Callable] = None,
    ):
        self.dataset = dataset  # an nlp dataset of snippets
        self.index_name = index_name
        self.embed_model = embed_model  # a model that needs to have an embed_query and embed_documents method
        # e.g. https://github.com/yjernite/transformers/blob/58e06cda0bf004ae84ab2381b1136814e0907c1d/examples/eli5/eli5_utils.py#L123
        self.embed_tokenizer = embed_tokenizer
        index_kwargs = index_kwargs if index_kwargs is not None else {}
        self.init_index(**index_kwargs)

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

    # TODO: batch version of query_dense_index
    #   https://github.com/yjernite/transformers/blob/58e06cda0bf004ae84ab2381b1136814e0907c1d/examples/eli5/eli5_utils.py#L530


class SparseIndexedDataset(BaseIndexedDataset):

    # indexes snippets with ElasticSearch
    def init_index(self, es_client, **kwargs):
        assert (
            _has_elasticsearch
        ), "You must install ElasticSearch to use SparseIndexedDataset. To do so you can run `pip install elasticsearch`"
        # Elasticsearch needs to be launched in another window, and a python client is declared withL
        # > es_client = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
        self.es_client = es_client
        self._make_es_index_snippets()
        # Here's the code for make_es_index_snippets:
        #   https://github.com/yjernite/transformers/blob/58e06cda0bf004ae84ab2381b1136814e0907c1d/examples/eli5/eli5_utils.py#L34

    def _make_es_index_snippets(self):
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
                yield passage

        # create the ES index
        for ok, action in es.helpers.streaming_bulk(client=es_client, index=index_name, actions=passage_generator(),):
            progress.update(1)
            successes += ok
        logger.info("Indexed %d documents" % (successes,))

    # send query to the elastic_search client, and return list of snippets
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


class DenseIndexedDataset(BaseIndexedDataset):
    def __init__(
        self,
        dataset: Dataset,
        index_name: str,
        embed_model: BaseEmbedModel,
        index_kwargs: Optional[dict] = None,
        embed_tokenizer: Optional[Callable] = None,
        batch_size=16,
        size=768,
    ):
        self.array_file_path = os.path.join(HF_INDEXES_CACHE, index_name)
        self.batch_size = batch_size
        self.size = size
        super().__init__(dataset, index_name, embed_model, index_kwargs, embed_tokenizer)

    # batch computes the snippet text representation with the model and add to memory-mapped numpy file
    # Let's start with Numpy, then investigate how pyarrow Tensors would work
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
        st_time = time()
        fp = np.memmap(
            self.array_file_path, dtype="float32", mode="w+", shape=(self.dataset.num_rows, self.size)
        )
        n_batches = math.ceil(self.dataset.num_rows / self.batch_size)
        for i in range(n_batches):
            texts = [
                p for p in self.dataset[i * self.batch_size : (i + 1) * self.batch_size]["text"]
            ]
            titles = [
                p for p in self.dataset[i * self.batch_size : (i + 1) * self.batch_size]["title"]
            ]
            reps = self.embed_model.embed_documents(texts, titles=titles, tokenizer=self.embed_tokenizer)
            fp[i * self.batch_size : (i + 1) * self.batch_size] = reps
            if i % 50 == 0:
                logger.info("Done writing batch={}/{},\ttime={:.2f}s".format(i + 1, n_batches, time() - st_time))
        fp.flush()
        del fp

    # load the numpy index into faiss
    # device is the index of the GPU, -1 for CPU
    def _load_dense_index(self, device=-1):
        fp = np.memmap(
            self.array_file_path, dtype="float32", mode="r", shape=(self.dataset.num_rows, self.size)
        )
        index_flat = faiss.IndexFlatIP(self.size)
        if device > -1:
            faiss_res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(faiss_res, device, index_flat)
        else:
            self.faiss_index = index_flat
        self.faiss_index.add(fp)
        fp.flush()

    # query the dense index either qith a text query or a pre-computed query vector
    def query_index(self, query, k=10, q_rep: Optional[np.array] = None):
        q_rep = self.embed_model.embed_queries([query], self.embed_tokenizer) if q_rep is None else q_rep
        scores, indices = self.faiss_index.search(q_rep, k)
        return scores[0], indices[0].astype(int)

    def query_index_batch(self, queries, k=10, q_rep: Optional[np.array] = None):
        q_rep = self.embed_model.embed_queries(queries, self.embed_tokenizer) if q_rep is None else q_rep
        scores, indices = self.faiss_index.search(q_rep, k)
        return scores, indices.astype(int)
