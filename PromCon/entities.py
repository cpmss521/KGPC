# -*- coding: utf-8 -*-
# @Time    : 2022/11/16 下午3:19
# @Author  : cp
# @File    : entities.py.py


from PromCon import sampling
from typing import List
from collections import OrderedDict
from torch.utils.data import Dataset as TorchDataset



class Token:
    def __init__(self, tid: int, index: int, span_start: int, span_end: int, phrase: str):
        self._tid = tid  # ID within the corresponding dataset
        self._index = index  # original token index in document

        self._span_start = span_start  # start of token span in document after Bert tokenization(inclusive)
        self._span_end = span_end  # end of token span in document after Bert tokenization (exclusive)+1
        self._phrase = phrase

    @property
    def index(self):
        return self._index## sentence token index position

    @property
    def span_start(self):
        return self._span_start

    @property
    def span_end(self):
        return self._span_end

    @property
    def span(self):
        return self._span_start, self._span_end

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Token):
            return self._tid == other._tid
        return False

    def __hash__(self):
        return hash(self._tid)

    def __str__(self):
        return self._phrase

    def __repr__(self):
        return self._phrase



class TokenSpan:
    def __init__(self, tokens):
        self._tokens = tokens

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def span_token(self):
        return self._tokens[0].index, self._tokens[-1].index # end of word in document without Bert tokenization(exclusive)

    def __getitem__(self, s):
        if isinstance(s, slice):
            return TokenSpan(self._tokens[s.start:s.stop:s.step])
        else:
            return self._tokens[s]

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)



class EntityType:
    def __init__(self, identifier, index, short_name, verbose_name):
        self._identifier = identifier
        self._index = index
        self._short_name = short_name
        self._verbose_name = verbose_name

    @property
    def identifier(self):
        return self._identifier

    @property
    def index(self):
        return self._index

    @property
    def short_name(self):
        return self._short_name

    @property
    def verbose_name(self):
        return self._verbose_name

    def __int__(self):
        return self._index

    def __eq__(self, other):
        if isinstance(other, EntityType):
            return self._identifier == other._identifier
        return False

    def __hash__(self):
        return hash(self._identifier)



class Entity:
    def __init__(self, eid: int, entity_type: EntityType, tokens: List[Token], phrase: str):

        self._eid = eid  # ID within the corresponding dataset
        self._entity_type = entity_type
        self._tokens = tokens
        self._phrase = phrase

    def as_tuple(self):
        # bert tokenizer position (end position exclusive)
        return self.span_start, self.span_end, self._entity_type

    def as_tuple_token(self):
        return self._tokens[0].index, self._tokens[-1].index, self._entity_type##end position  [inclusive]

    @property
    def entity_type(self):
        return self._entity_type

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def span_start(self):
        #bert 分词之后的span开始位置
        return self._tokens[0].span_start

    @property
    def span_end(self):
        # bert 分词之后的span结束位置
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def span_token(self):
        # 原始句子的span开始和结束(inclusive)位置
        return self._tokens[0].index, self._tokens[-1].index

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self._eid == other._eid
        return False

    def __hash__(self):
        return hash(self._eid)

    def __str__(self):
        return self._phrase




class Document:
    def __init__(self, doc_id: int, tokens: List[Token], entities: List[Entity],
                 encoding: List[int], boundary: List[int], query_length:int,type_id:int):
        self._doc_id = doc_id  # ID within the corresponding dataset

        self._tokens = tokens # Token object list
        self._entities = entities

        # byte-pair document encoding including special tokens ([CLS] and [SEP])
        self._encoding = encoding
        self._boundary= boundary
        self._query_length = query_length
        self._type_id = type_id


    @property
    def doc_id(self):
        return self._doc_id

    @property
    def entities(self):
        return self._entities

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def encoding(self):
        return self._encoding

    @encoding.setter
    def encoding(self, value):
        self._encoding = value

    @property
    def boundary(self):
        return self._boundary

    @boundary.setter
    def boundary(self, value):
        self._boundary = value

    @property
    def query_length(self):
        return self._query_length

    @query_length.setter
    def query_length(self, value):
        self._query_length = value

    @property
    def type_id(self):
        return self._type_id

    @type_id.setter
    def type_id(self, value):
        self._type_id = value

    def __eq__(self, other):
        if isinstance(other, Document):
            return self._doc_id == other._doc_id
        return False

    def __hash__(self):
        return hash(self._doc_id)





class Dataset(TorchDataset):
    TRAIN_MODE = 'train'
    EVAL_MODE = 'eval'

    def __init__(self, label):
        self._label = label
        self._mode = Dataset.TRAIN_MODE
        self._documents = OrderedDict()## sentence index: sentence
        self._entities = OrderedDict()

        # current ids
        self._doc_id = 0
        self._eid = 0
        self._tid = 0

    def create_token(self, idx, span_start, span_end, phrase) -> Token:
        # idx: token position index in sentence
        token = Token(self._tid, idx, span_start, span_end, phrase)
        self._tid += 1
        return token

    def create_document(self, tokens, entity_mentions, doc_encoding, boundary, query_length,type_id) -> Document:
        document = Document(self._doc_id, tokens, entity_mentions, doc_encoding, boundary, query_length,type_id)
        self._documents[self._doc_id] = document
        self._doc_id += 1

        return document

    def create_entity(self, entity_type, tokens, phrase) -> Entity:
        mention = Entity(self._eid, entity_type, tokens, phrase)
        self._entities[self._eid] = mention
        self._eid += 1
        return mention

    def __len__(self):
        return len(self._documents)

    def __getitem__(self, index: int):
        doc = self._documents[index]

        if self._mode == Dataset.TRAIN_MODE:

            return sampling.train_sample(doc)
        else:
            return sampling.eval_sample(doc)

    def switch_mode(self, mode):
        self._mode = mode

    @property
    def label(self):
        return self._label

    @property
    def input_reader(self):
        return self._input_reader

    @property
    def documents(self):
        return list(self._documents.values())

    @property
    def entities(self):
        return list(self._entities.values())

    @property
    def document_count(self):
        return len(self._documents)

    @property
    def entity_count(self):
        return len(self._entities)
