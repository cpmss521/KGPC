# -*- coding: utf-8 -*-
# @Time    : 2022/09/16 上午10:58
# @Author  : cp
# @File    : main.py

import warnings
warnings.filterwarnings("ignore")
import argparse
from args import train_argparser, eval_argparser
from PromCon.inputReader import JsonInputReader
from PromCon.PromConTrainer import PromConTrainer
from config_reader import process_configs


def _train():
    arg_parser = train_argparser()
    process_configs(target=__train, arg_parser=arg_parser)


def __train(run_args):
    trainer = PromConTrainer(run_args)
    trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                  types_path=run_args.types_path, input_reader_cls=JsonInputReader)


def _eval():
    arg_parser = eval_argparser()
    process_configs(target=__eval, arg_parser=arg_parser)

def __eval(run_args):
    trainer = PromConTrainer(run_args)
    trainer.eval(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                 input_reader_cls=JsonInputReader)




if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'eval'")
    args, _ = arg_parser.parse_known_args()


    if args.mode == 'train':
        _train()
    elif args.mode == 'eval':
        _eval()
    else:
        raise Exception("Mode not in ['train', 'eval'], e.g. 'python enRelTsg.py train ...'")
