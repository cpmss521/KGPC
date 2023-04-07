# -*- coding: utf-8 -*-
# @Time    : 2022/11/16 下午9:45
# @Author  : cp
# @File    : evaluator.py.py


import json
import torch
from typing import List
from seqeval.metrics import f1_score,precision_score,recall_score

from PromCon.entities import Document, Dataset
from PromCon.inputReader import BaseInputReader
import numpy as np


class Evaluator:
    def __init__(self, dataset: Dataset, input_reader: BaseInputReader, predictions_path: str, epoch: int, dataset_label: str):

        self._dataset = dataset
        self._input_reader = input_reader
        self._predictions_path = predictions_path
        self._epoch = epoch
        self._dataset_label = dataset_label
        self._bound_map = self._input_reader.bound_map
        self._gt_entities = []  # all example ground truth [[(s,e,type),(s,e,type)],[],[]..]
        self._pred_entities = []  # prediction

        self._convert_gt(self._dataset.documents)



    def _convert_gt(self, docs: List[Document]):
        for doc in docs:
            gt_boundary = doc.boundary
            gt_type_id = doc.type_id
            sample_gt_entities = self.extract_entities(gt_boundary, self._bound_map , gt_type_id)#BIO label sequence

            self._gt_entities.append(sample_gt_entities)

    def extract_entities(self, boundary_id, label_dict, type_id):
        """" extract entity sequence for each sentence """

        id2label = {v:k for k,v in label_dict.items()}
        bd_label = [id2label[i] for i in boundary_id] ## BIO label sequence
        entity_name = self._input_reader._idx2entity_type[type_id].short_name
        new_bd_label = ["O"]*len(bd_label)
        for index, item in enumerate(bd_label):
            if item =="B":
                new_bd_label[index]="B-"+entity_name
            elif item == "I":
                if index - 1 >= 0 and bd_label[index - 1] == "O":
                    new_bd_label[index] = "B-" + entity_name
                    # new_bd_label[index-1] = "B-" + entity_name
                    # new_bd_label[index] = "I-" + entity_name
                else:
                    new_bd_label[index] = "I-" + entity_name

        return new_bd_label



    def eval_batch(self, bound_logist: torch.tensor, entity_type_id):
        ## entity_type_id: entity in each sentence

        logits = bound_logist.detach().cpu().numpy()
        pred_lst = np.argmax(logits, axis=-1)  # B,S [label index]
        for bound_list, type_id in zip(pred_lst, entity_type_id):
            pred_entity = self.extract_entities(bound_list, self._bound_map , type_id)
            self._pred_entities.append(pred_entity)


    def compute_scores(self):
        print("Evaluation")

        print("")
        print("--- Entities (named entity recognition (NER)) ---")
        print("An entity is considered correct if the entity type and span is predicted correctly")
        print("")
        ner_f1 = f1_score(self._gt_entities, self._pred_entities)
        ner_recall = recall_score(self._gt_entities, self._pred_entities)
        ner_precision = precision_score(self._gt_entities, self._pred_entities)

        return ner_precision, ner_recall, ner_f1

    def get_entity_span(self, label_lst, enty_type):
        entities = []
        temp_entity = []

        for idx, label in enumerate(label_lst):
            if label.split('-')[0] == "B":
                if len(temp_entity) > 0:
                    entities.append((temp_entity[0], temp_entity[-1], enty_type))
                    temp_entity = []
                temp_entity.append(idx)
            elif label.split("-")[0] == "I":
                temp_entity.append(idx)
            elif label.split('-')[0] == "O" and len(temp_entity) > 0:
                entities.append((temp_entity[0], temp_entity[-1], enty_type))
                temp_entity = []

        if len(temp_entity) > 0:
            entities.append((temp_entity[0], temp_entity[-1], enty_type))

        return entities

    def store_predictions(self):
        predictions = []

        for i, doc in enumerate(self._dataset.documents):
            tokens = doc.tokens
            type_id = doc.type_id
            entity_name = self._input_reader._idx2entity_type[type_id].short_name

            pred_entities = self._pred_entities[i]## sequence label
            span = self.get_entity_span(pred_entities, entity_name)#[(s,e,type),..()]
            doc_predictions = dict(tokens=[t.phrase for t in tokens], entities=span)
            predictions.append(doc_predictions)

        # store as json
        label, epoch = self._dataset_label, self._epoch
        with open(self._predictions_path % (label, epoch), 'w') as predictions_file:
            json.dump(predictions, predictions_file)

