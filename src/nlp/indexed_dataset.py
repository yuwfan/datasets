import logging
import math
import os
from time import time
from typing import Dict, List, Tuple, Union

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


class BaseIndexedDataset:
    def __init__(
        self,
        dataset: Dataset,
        split_function=None,
        split_kwargs=None,
        index_kwargs=None,
        embed_model=None,
        embed_tokenizer=None,
    ):
        self.dataset = dataset  # an nlp dataset
        self.dataset_snippets = self.make_snippets(split_function, **split_kwargs)
        self.embed_model = embed_model  # a model that needs to have an embed_query and embed_documents method
        # e.g. https://github.com/yjernite/transformers/blob/58e06cda0bf004ae84ab2381b1136814e0907c1d/examples/eli5/eli5_utils.py#L123
        self.embed_tokenizer = embed_tokenizer
        self.init_index(**index_kwargs)

    def __getitem__(self, key: Union[int, slice, str]) -> Union[Dict, List]:
        return self.dataset_snippets[key]

    def query_index(self, query, k: int = 10) -> Tuple[List[float], List[int]]:
        raise NotImplementedError

    def init_index(self):
        raise NotImplementedError

    def query_index_batch(self, queries, k: int = 10) -> Tuple[List[List[float]], List[List[int]]]:
        raise NotImplementedError

    # TODO: batch version of query_dense_index
    #   https://github.com/yjernite/transformers/blob/58e06cda0bf004ae84ab2381b1136814e0907c1d/examples/eli5/eli5_utils.py#L530

    # we need to split the dataset items (e.g. Wikipedia articles) into fixed sized
    # passages / snippets to index
    def make_snippets(self, split_func, **split_args) -> Dataset:
        return self.dataset
        # TODO: implement
        # My first implementation makes an NLP dataset of snippets:
        #   https://github.com/yjernite/transformers/blob/eli5_examples/examples/eli5/data/wiki_snippets/wiki_snippets.py
        # here are the split functions for Wikipedia and Wiki40b:
        #   https://github.com/yjernite/transformers/blob/58e06cda0bf004ae84ab2381b1136814e0907c1d/examples/eli5/data/wiki_snippets/wiki_snippets.py#L89
        #   https://github.com/yjernite/transformers/blob/58e06cda0bf004ae84ab2381b1136814e0907c1d/examples/eli5/data/wiki_snippets/wiki_snippets.py#L59
        # each snippet has the text, article title, titles of sections by the text, and provenance info to map back to self.dataset items


class SparseIndexedDataset(BaseIndexedDataset):

    # indexes snippets with ElasticSearch
    def init_index(self, es_client: es.Elasticsearch, index_name):
        assert (
            _has_elasticsearch
        ), "You must install ElasticSearch to use SparseIndexedDataset. To do so you can run `pip install elasticsearch`"
        # Elasticsearch needs to be launched in another window, and a python client is declared withL
        # > es_client = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
        self.es_client = es_client
        self.es_index_name = index_name
        self._make_es_index_snippets()
        # Here's the code for make_es_index_snippets:
        #   https://github.com/yjernite/transformers/blob/58e06cda0bf004ae84ab2381b1136814e0907c1d/examples/eli5/eli5_utils.py#L34

    def make_es_index_snippets(self):
        from tqdm.auto import tqdm

        es_client = self.es_client
        passages_dset = self.dataset_snippets
        index_name = self.dataset.info.builder_name  # TODO: use full name
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
            index=self.es_index_name,
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
        split_function=None,
        split_kwargs=None,
        index_kwargs=None,
        embed_model=None,
        embed_tokenizer=None,
        batch_size=10_000,
        size=128,
    ):
        self.array_file_path = os.path.join(HF_INDEXES_CACHE, dataset.info.builder_name)
        super().__init__(dataset, split_function, split_kwargs, index_kwargs, embed_model, embed_tokenizer)
        self.batch_size = batch_size
        self.size = size

    # batch computes the snippet text representation with the model and add to memory-mapped numpy file
    # Let's start with Numpy, then investigate how pyarrow Tensors would work
    def init_index(self, device=-1):
        assert (
            _has_faiss
        ), "You must install Faiss to use DenseIndexedDataset. To do so you can run `pip install faiss`"
        if not os.path.exists(self.array_file_path):
            self._build_dense_index()
        self._load_dense_index(device)

    def _build_dense_index(self):
        st_time = time()
        logger.info("Building dense index for dataset {}.".format(self.dataset.info.builder_name))
        fp = np.memmap(
            self.array_file_path, dtype="float32", mode="w+", shape=(self.dataset_snippets.num_rows, self.size)
        )
        n_batches = math.ceil(self.dataset_snippets.num_rows / self.batch_size)
        for i in range(n_batches):
            passages = [
                p for p in self.dataset_snippets[i * self.batch_size : (i + 1) * self.batch_size]["passage_text"]
            ]
            reps = self.embed_model.embed_documents(passages, self.embed_tokenizer)
            fp[i * self.batch_size : (i + 1) * self.batch_size] = reps
            if i % 50 == 0:
                logger.info(i, time() - st_time)
        fp.close()

    # load the numpy index into faiss
    # device is the index of the GPU, -1 for CPU
    def _load_dense_index(self, device=-1):
        fp = np.memmap(
            self.array_file_path, dtype="float32", mode="r", shape=(self.dataset_snippets.num_rows, self.size)
        )
        faiss_res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatIP(self.size)
        if device > -1:
            self.faiss_index = faiss.index_cpu_to_gpu(faiss_res, device, index_flat)
        else:
            self.faiss_index = index_flat
        self.faiss_index.add(fp)
        fp.close()

    # query the dense index either qith a text query or a pre-computed query vector
    def query_index(self, query, k=10):
        q_rep = self.embed_model.embed_query(query, self.embed_tokenizer)
        scores, indices = self.fais_index.search(q_rep, self.k)
        return scores[0], indices[0]
