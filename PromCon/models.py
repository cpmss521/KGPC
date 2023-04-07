# -*- coding: utf-8 -*-
# @Time    : 2022/11/16 下午2:57
# @Author  : cp
# @File    : models.py


import torch
from typing import List
from torch import nn as nn
import torch.nn.functional as F
from transformers import BertConfig,BertModel,BertPreTrainedModel
from PromCon.utils import get_positive_expectation, get_negative_expectation


class PromContrast(BertPreTrainedModel):
    VERSION = '1.0'

    def __init__(self, config: BertConfig, bert_dropout:float,freeze_transformer: bool, device, pool_type:str = "max",output_dim:int=3):
        super(PromContrast, self).__init__(config)

        self.pool_type = pool_type
        self._bound_criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)
        # BERT model
        self.bert = BertModel(config)
        self.bert_dropout = nn.Dropout(bert_dropout)
        self.boundary_classifier = nn.Linear(config.hidden_size,output_dim)

        if freeze_transformer:
            print("Freeze transformer weights")
            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    def combine(self, sub, sup_mask, pool_type = "max" ):
        sup = None
        if len(sub.shape) == len(sup_mask.shape) :
            if pool_type == "mean":
                size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.sum(dim=2) / size
            if pool_type == "sum":
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.sum(dim=2)
            if pool_type == "max":
                m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
                sup = m + sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.max(dim=2)[0]
                sup[sup==-1e30]=0
        else:
            if pool_type == "mean":
                size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub
                sup = sup.sum(dim=2) / size
            if pool_type == "sum":
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub
                sup = sup.sum(dim=2)
            if pool_type == "max":
                m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
                sup = m + sub
                sup = sup.max(dim=2)[0]
                sup[sup==-1e30]=0
        return sup

    def sequence_query_loss(self, sequence, query, match_labels, mask, measure='JSD'):
        """
            SQ-CL module
            Computing the Contrastive Loss between the sequence and query.
            :param video: sequence rep (B, S, H)
            :param query: query prompt rep (B, H)
            :param match_labels: match labels (B, S)
            :param mask: mask (B, S)
            :param measure: estimator of the mutual information
        """
        # generate mask
        pos_mask = match_labels.type(torch.float32)  # (B, S)
        neg_mask = (torch.ones_like(pos_mask) - pos_mask) * mask  # (B, S)

        # compute scores
        query = query.unsqueeze(2)  # (B, H, 1)
        res = torch.matmul(sequence, query).squeeze(2)  # (B, S)

        # computing expectation for the MI between the target moment (positive samples) and query.
        E_pos = get_positive_expectation(res * pos_mask, measure, average=False)
        E_pos = torch.sum(E_pos, dim=1) / (torch.sum(pos_mask, dim=1) + 1e-12)  # (B, )

        # computing expectation for the MI between clips except target moment (negative samples) and query.
        E_neg = get_negative_expectation(res * neg_mask, measure, average=False)
        E_neg = torch.sum(E_neg, dim=1) / (torch.sum(neg_mask, dim=1) + 1e-12)  # (B, )

        E = E_neg - E_pos  # (B, )
        return torch.mean(E)

    def forward(self, encodings: torch.tensor, segment_ids:torch.tensor, context_masks: torch.tensor,token_masks: torch.tensor,
                        token_masks_bool:torch.tensor, match_label:torch.tensor, query_len:List,boundary:torch.tensor=None):

        # get contextualized token embeddings from last transformer layer
        sequence_output = self.bert(input_ids=encodings,token_type_ids=segment_ids,attention_mask =context_masks.float())[0]
        # get query encoding and mask
        batch_size = sequence_output.size()[0]
        max_query_len = max(query_len) 
        query_prompt_encoding = torch.zeros((batch_size, max_query_len, sequence_output.size()[-1])).to(sequence_output.device)   
        query_mask = torch.zeros((batch_size, max_query_len), dtype=torch.bool).to(sequence_output.device) 

        for i in range(batch_size):
            query_prompt_encoding[i, :query_len[i]] = sequence_output[i, -(query_len[i]+1):-1, ]
            query_mask[i, :query_len[i]] = torch.ones(query_len[i], dtype=torch.bool)

        query_pooling = torch.sum(query_prompt_encoding*query_mask.unsqueeze(-1), dim=1) / torch.sum(query_mask.unsqueeze(-1), dim=1) 
        # context representation
        sequence_output = self.bert_dropout(sequence_output)  
        h_token = self.combine(sequence_output, token_masks, self.pool_type)   
        bound_prob = self.boundary_classifier(h_token) 

        if boundary is not None:
            active_loss = token_masks_bool.view(-1) == 1
            boundary_loss = self._bound_criterion(bound_prob.view(-1,bound_prob.shape[-1])[active_loss], boundary.view(-1)[active_loss])
            sq_cl = self.sequence_query_loss(h_token, query_pooling, match_label, token_masks_bool)

            loss = boundary_loss + sq_cl
            
            return loss
        else:
            bound_prob = F.log_softmax(bound_prob, dim=2)# B*S

            return bound_prob

# Model access

_MODELS = {
    'PromContrast': PromContrast,
}


def get_model(name):
    return _MODELS[name]

