# -*- coding: utf-8 -*-
# @Time    : 2022/11/16 下午3:08
# @Author  : cp
# @File    : PromConTrainer.py

import argparse
import math
import os
from tqdm import tqdm
from typing import Type
import torch
from torch.nn import DataParallel
from torch.optim import Optimizer
import transformers
from torch.utils.data import DataLoader
from transformers import AdamW, BertConfig
from transformers import BertTokenizer

from PromCon import models
from PromCon import sampling
from PromCon import utils
from PromCon.entities import Dataset
from PromCon.evaluator import Evaluator
from PromCon.inputReader import JsonInputReader, BaseInputReader
from PromCon.loss import BoundaryLoss, Loss
from PromCon.baseTrainer import BaseTrainer





class PromConTrainer(BaseTrainer):
    """ few-shot learning entity extraction training and evaluation """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # byte-pair encoding
        self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,do_lower_case=args.lowercase,cache_dir=args.cache_path)

    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: Type[BaseInputReader]):
        args = self._args
        train_label, valid_label = 'train', 'valid'

        # create log csv files
        self._init_train_logging(train_label)
        self._init_eval_logging(valid_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, self._logger)
        train_dataset = input_reader.read(train_path, train_label)
        valid_dataset = input_reader.read(valid_path, valid_label)

        self._log_datasets(input_reader)

        train_sample_count = train_dataset.document_count
        updates_epoch = train_sample_count // args.train_batch_size
        updates_total = updates_epoch * args.epochs

        self._logger.info("Updates per epoch: %s" % updates_epoch)
        self._logger.info("Updates total: %s" % updates_total)


        # load model
        model = self._load_model()

        # KGPC is currently optimized on a single GPU and not thoroughly tested in a multi GPU setup
        # If you still want to train KGPC on multiple GPUs, uncomment the following lines
        # # parallelize model
        # if self._device.type != 'cpu':
        #     model = torch.nn.DataParallel(model)

        model.to(self._device)

        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
        # create scheduler
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=args.lr_warmup * updates_total,
                                                                 num_training_steps=updates_total)

        compute_loss = BoundaryLoss(model, optimizer, scheduler, args.max_grad_norm)


        # train model
        best_f1 = 0.0
        for epoch in range(args.epochs):
            # train epoch
            self._train_epoch(model, compute_loss, optimizer, train_dataset, updates_epoch, epoch)

            # eval valid data under last epoch
            if not args.final_eval or (epoch == args.epochs - 1):
                entity_eval =  self._eval(model, valid_dataset, input_reader, epoch + 1)
                if best_f1 < entity_eval[2]:
                    ## entity_eval[2]: micro F1 score
                    self._logger.info("Best F1 score update, from {:.4f} to {:.4f}".format(best_f1, entity_eval[2]))
                    # save final model
                    best_f1 = entity_eval[2]
                    extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
                    global_iteration = args.epochs * updates_epoch
                    self._save_model(self._save_path, model, self._tokenizer, global_iteration,
                                     optimizer=optimizer if self._args.save_optimizer else None, save_as_best=True,
                                     extra=extra, include_iteration=False)
                else:
                    self._logger.info("Best F1 score not changed, is still {:.4f}".format(best_f1))

        self._logger.info("Best F1 score is {:.4f}".format(best_f1))
        self._logger.info("Logged in: %s" % self._log_path)
        self._logger.info("Saved in: %s" % self._save_path)
        self._close_summary_writer()



    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss, optimizer: Optimizer, dataset: Dataset,
                     updates_epoch: int, epoch: int):
        self._logger.info("Train epoch: %s" % epoch)

        # create data loader
        dataset.switch_mode(Dataset.TRAIN_MODE)
        data_loader = DataLoader(dataset, batch_size=self._args.train_batch_size, shuffle=True, drop_last=False,
                                 num_workers=self._args.sampling_processes, collate_fn= sampling.collate_fn_padding)

        model.zero_grad()

        iteration = 0
        total = dataset.document_count // self._args.train_batch_size
        for batch in tqdm(data_loader, total=total, desc='Train epoch %s' % epoch):
            model.train()
            batch = utils.to_device(batch, self._device)

            # forward step
            loss= model(encodings=batch['encodings'], segment_ids=batch['segment_ids'],context_masks=batch['context_masks'],
                        token_masks=batch['token_masks'],token_masks_bool=batch['token_masks_bool'],match_label=batch['match_label'],
                        query_len=batch['query_len'],boundary=batch['boundary'])

            # compute loss and optimize parameters
            batch_loss = compute_loss.compute(loss=loss)
            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % self._args.train_log_iter == 0:
                self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)

        return iteration

    def _eval(self, model: torch.nn.Module, dataset: Dataset, input_reader: BaseInputReader,epoch: int = 0):
        self._logger.info("Evaluate: %s" % dataset.label)


        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module

        # create evaluator
        predictions_path = os.path.join(self._log_path, 'predictions_%s_epoch_%s.json')
        evaluator = Evaluator(dataset, input_reader, predictions_path, epoch, dataset.label)

        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self._args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=self._args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self._args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
                # move batch to selected device
                batch = utils.to_device(batch, self._device)

                # run model (forward pass)
                bound_log = model(encodings=batch['encodings'], segment_ids=batch['segment_ids'],context_masks=batch['context_masks'],
                        token_masks=batch['token_masks'],token_masks_bool=batch['token_masks_bool'],match_label=batch['match_label'],
                        query_len=batch['query_len'])

                # evaluate batch
                evaluator.eval_batch(bound_log, batch['entity_type_id'])

        ner_eval = evaluator.compute_scores()
        self._logger.info("P = {:.4f},R = {:.4f},F1 = {:.4f}".format(ner_eval[0],ner_eval[1],ner_eval[2]))
        if self._args.store_predictions and dataset.label=='test':
            evaluator.store_predictions()

        return ner_eval

    def eval(self, dataset_path: str, types_path: str, input_reader_cls: Type[BaseInputReader]):
        args = self._args
        dataset_label = 'test'

        self._logger.info("Dataset: %s" % dataset_path)
        self._logger.info("Model: %s" % args.model_type)

        # create log csv files
        self._init_eval_logging(dataset_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer,logger=self._logger)
        test_dataset = input_reader.read(dataset_path, dataset_label)
        self._log_datasets(input_reader)

        # load model
        model = self._load_model()
        model.to(self._device)

        # evaluate
        self._eval(model, test_dataset, input_reader)
        self._logger.info("Logged in: %s" % self._log_path)
        self._close_summary_writer()


    def _load_model(self):
        model_class = models.get_model(self._args.model_type)

        config = BertConfig.from_pretrained(self._args.model_path, cache_dir=self._args.cache_path)

        config.model_version = model_class.VERSION
        model = model_class.from_pretrained(self._args.model_path,
                                            config=config,
                                            # model parameters
                                            bert_dropout = self._args.bert_dropout,
                                            freeze_transformer=self._args.freeze_transformer,
                                            device=self._device,
                                            pool_type=self._args.pool_type)

        return model

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.00}]

        return optimizer_params

    def _log_datasets(self, input_reader):
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)
        self._logger.info("Context size: %s" % input_reader.context_size)## max sent len after bert

        self._logger.info("Entities:")
        for e in input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            self._logger.info("Entity count: %s" % d.entity_count)



    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self._args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss, global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss, global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)



    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,

                  epoch: int, iteration: int, global_iteration: int, label: str):

        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,
                      epoch, iteration, global_iteration)


    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})


    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'rel_prec_micro', 'rel_rec_micro', 'rel_f1_micro',
                                                 'epoch', 'iteration', 'global_iteration']})



