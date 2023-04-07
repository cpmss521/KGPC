# -*- coding: utf-8 -*-
# @Time    : 2022/11/16 下午4:29
# @Author  : cp
# @File    : sampling.py

import torch
from PromCon import utils


def train_sample(doc):

    encodings = doc.encoding
    boundary = doc.boundary
    token_count = len(doc.tokens)
    context_size = len(encodings) # [CLS] + context + [SEP] + query + [SEP]
    segment_ids = [0] * (context_size - doc.query_length -1) + [1] * (doc.query_length + 1)

    token_masks = []
    for t in doc.tokens:
        token_masks.append(create_entity_mask(*t.span, context_size))
    # create tensors
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    context_masks = torch.ones(context_size, dtype=torch.bool)
    token_masks = torch.stack(token_masks)
    token_masks_bool = torch.ones(token_count, dtype=torch.bool)

    # construct moment localization foreground and background labels
    match_label = [1 if bd > 0 else 0 for bd in boundary]


    return dict(encodings=encodings,
                segment_ids =segment_ids,
                context_masks=context_masks,
                token_masks=token_masks,
                token_masks_bool=token_masks_bool,
                query_len = doc.query_length,
                match_label=torch.tensor(match_label,dtype=torch.int),
                boundary=torch.tensor(boundary,dtype=torch.long)
                )


def eval_sample(doc):

    encodings = doc.encoding
    boundary = doc.boundary
    token_count = len(doc.tokens)
    context_size = len(encodings) # [CLS] + context + [SEP] + query + [SEP]
    segment_ids = [0] * (context_size - doc.query_length -1) + [1] * (doc.query_length + 1)

    token_masks = []
    for t in doc.tokens:
        token_masks.append(create_entity_mask(*t.span, context_size))
    # create tensors
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    context_masks = torch.ones(context_size, dtype=torch.bool)
    token_masks = torch.stack(token_masks)
    token_masks_bool = torch.ones(token_count, dtype=torch.bool)

    # construct moment localization foreground and background labels
    match_label = [1 if bd > 0 else 0 for bd in boundary]

    entity_type_id = doc.type_id

    return dict(encodings=encodings,
                segment_ids =segment_ids,
                context_masks=context_masks,
                token_masks=token_masks,
                token_masks_bool=token_masks_bool,
                query_len = doc.query_length,
                match_label=torch.tensor(match_label,dtype=torch.int),
                entity_type_id=entity_type_id
                )


def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        if key == 'query_len':
            padded_batch[key] = []
            for s in batch:padded_batch[key].append(s[key])
        elif key == 'entity_type_id':
            padded_batch[key] = []
            for s in batch:padded_batch[key].append(s[key])
        else:
            padded_batch[key] = utils.padded_stack([s[key] for s in batch])

    return padded_batch





def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask



