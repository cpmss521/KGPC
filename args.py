# -*- coding: utf-8 -*-
# @Time    : 2022/11/16 上午10:59
# @Author  : cp
# @File    : args.py.py

import json
import argparse



def _add_common_args(arg_parser):
    arg_parser.add_argument('--config', type=str)

    # Input
    arg_parser.add_argument('--types_path', type=str, help="Path to type specifications")

    # Preprocessing
    arg_parser.add_argument('--tokenizer_path', type=str, help="Path to tokenizer")
    arg_parser.add_argument('--lowercase', action='store_true', default=False,help="If true, input is lowercased during preprocessing")
    arg_parser.add_argument('--sampling_processes', type=int, default=2,help="Number of sampling processes. 0 = no multiprocessing for sampling")
    arg_parser.add_argument('--cache_path', type=str, default=None,help="Path to cache transformer models (for HuggingFace transformers library)")

    # Logging
    arg_parser.add_argument('--label', type=str, help="Label of run. Used as the directory name of logs/models")
    arg_parser.add_argument('--log_path', type=str, help="Path do directory where training/evaluation logs are stored")
    arg_parser.add_argument('--store_predictions', action='store_true', default=False,help="If true, store predictions on disc (in log directory)")

    # Model / Training / Evaluation
    arg_parser.add_argument('--seed', type=int, default=58986, help="Seed")  ## 58986, 3047,
    arg_parser.add_argument('--model_path', type=str, help="Path to directory that contains model checkpoints")
    arg_parser.add_argument('--model_type', type=str, default="EnRelTSG", help="Type of model")
    arg_parser.add_argument('--eval_batch_size', type=int, default=1, help="Evaluation batch size")
    arg_parser.add_argument('--freeze_transformer', action='store_true', default=False, help="Freeze BERT weights")
    arg_parser.add_argument('--pool_type', type=str, default="max")
    arg_parser.add_argument('--bert_dropout', type=float, default=0.1)








def train_argparser():
    arg_parser = argparse.ArgumentParser()

    # Input
    arg_parser.add_argument('--train_path', type=str, help="Path to train dataset")
    arg_parser.add_argument('--valid_path', type=str, help="Path to validation dataset")

    # Logging
    arg_parser.add_argument('--save_path', type=str, help="Path to directory where model checkpoints are stored")
    arg_parser.add_argument('--save_optimizer', action='store_true', default=False,help="Save optimizer alongside model")
    arg_parser.add_argument('--train_log_iter', type=int, default=1, help="Log training process every x iterations")
    arg_parser.add_argument('--final_eval', action='store_true', default=False,help="Evaluate the model only after training, not at every epoch")

    # Model / Training
    arg_parser.add_argument('--train_batch_size', type=int, default=16, help="Training batch size")
    arg_parser.add_argument('--epochs', type=int, default=20, help="Number of epochs")
    arg_parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    arg_parser.add_argument('--lr_warmup', type=float, default=0.1, help="Proportion of total train iterations to warmup in linear increase/decrease schedule")
    arg_parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay to apply")
    arg_parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm")



    _add_common_args(arg_parser)

    return arg_parser




def eval_argparser():
    arg_parser = argparse.ArgumentParser()

    # Input
    arg_parser.add_argument('--dataset_path', type=str, help="Path to dataset")

    _add_common_args(arg_parser)

    return arg_parser



