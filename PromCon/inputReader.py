# -*- coding: utf-8 -*-
# @Time    : 2022/11/16 下午3:16
# @Author  : cp
# @File    : inputReader.py



import json
from tqdm import tqdm
from abc import abstractmethod, ABC
from collections import OrderedDict
from logging import Logger
from typing import List, Iterable
from transformers import BertTokenizer
from PromCon.entities import EntityType, Dataset, Document, Entity




class BaseInputReader(ABC):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, logger: Logger = None):

        types = json.load(open(types_path), object_pairs_hook=OrderedDict)

        self._entity_types = OrderedDict()
        self._idx2entity_type = OrderedDict()

        # entities
        # add 'None' entity type
        none_entity_type = EntityType('None', 0, 'None', 'No Entity')
        self._entity_types['None'] = none_entity_type
        self._idx2entity_type[0] = none_entity_type

        # specified entity types
        for i, (key, v) in enumerate(types['entities'].items()):
            entity_type = EntityType(key, i + 1, v['short'], v['verbose'])
            self._entity_types[key] = entity_type
            self._idx2entity_type[i + 1] = entity_type

        # boundary map
        self.bound_map = types['bound_map']## BIOES dict

        self._datasets = dict()
        self._tokenizer = tokenizer
        self._logger = logger
        self._vocabulary_size = tokenizer.vocab_size
        self._context_size = -1


    @abstractmethod
    def read(self, dataset_path, dataset_label):
        pass

    def get_dataset(self, label) -> Dataset:
        return self._datasets[label]

    def get_entity_type(self, idx) -> EntityType:
        entity = self._idx2entity_type[idx]
        return entity

    def _calc_context_size(self, datasets: Iterable[Dataset]):
        sizes = []

        for dataset in datasets:
            for doc in dataset.documents:
                sizes.append(len(doc.encoding))

        context_size = max(sizes)
        return context_size

    @property
    def datasets(self):
        return self._datasets

    @property
    def entity_types(self):
        return self._entity_types

    @property
    def entity_type_count(self):
        return len(self._entity_types)

    @property
    def vocabulary_size(self):
        return self._vocabulary_size

    @property
    def context_size(self):
        return self._context_size

    def __str__(self):
        string = ""
        for dataset in self._datasets.values():
            string += "Dataset: %s\n" % dataset
            string += str(dataset)

        return string

    def __repr__(self):
        return self.__str__()


class JsonInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, logger: Logger = None):
        super().__init__(types_path, tokenizer, logger)


    def read(self, dataset_path, dataset_label):
        dataset = Dataset(dataset_label)
        self._parse_dataset(dataset_path, dataset)
        self._datasets[dataset_label] = dataset
        self._context_size = self._calc_context_size(self._datasets.values())

        return dataset

    def _parse_dataset(self, dataset_path, dataset):

        if 'jsonl' in dataset_path:
            documents = []
            with open(dataset_path,"r",encoding="utf8") as f:
                for line in f:
                    example_json = json.loads(line)
                    documents.append(example_json)
        else:
            documents = json.load(open(dataset_path, 'r'))

        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset.label):
            self._parse_document(document, dataset)

    def _parse_document(self, doc, dataset) -> Document:
        sentence = doc['context'].split(" ")
        query_prompt = doc['query']
        boundary = doc["boundary"]
        span = doc['span_position']
        enty_type = doc['type']

        # parse tokens
        doc_tokens, doc_encoding, query_len = self._parse_tokens(sentence, query_prompt, dataset) 

        ## parse boundary map
        boundary = [self.bound_map.get(item) for item in boundary]

        # parse entity mentions
        entities = self._parse_entities(span, enty_type, doc_tokens, dataset)

        # create document
        document = dataset.create_document(doc_tokens, entities, doc_encoding, boundary, query_len,self._entity_types[enty_type].index)

        return document


    def _parse_tokens(self, jtokens, query_prompt, dataset):
        doc_tokens = []  # save Token object

        # [CLS] +query_prompt +[SEP] + sentence + [SEP]
        doc_encoding = [self._tokenizer.convert_tokens_to_ids('[CLS]')]

        # parse tokens
        for i, token_phrase in enumerate(jtokens):
            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
            span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))
            token = dataset.create_token(i, span_start, span_end, token_phrase)
            doc_tokens.append(token)
            doc_encoding += token_encoding
        doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]
        query_id = self._tokenizer.encode(query_prompt, add_special_tokens=False)
        doc_encoding += query_id
        doc_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]

        return doc_tokens, doc_encoding, len(query_id)

    def _parse_entities(self, jentities, enty_type, doc_tokens, dataset) -> List[Entity]:

        entities = []
        for jentity in jentities:
            entity_type = self._entity_types[enty_type]
            start, end = jentity[0], jentity[1]# end position( exclusive)

            # create entity mention
            tokens = doc_tokens[start:end]
            phrase = " ".join([t.phrase for t in tokens])
            entity = dataset.create_entity(entity_type, tokens, phrase)
            entities.append(entity)

        return entities



