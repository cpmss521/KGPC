# -*- coding: utf-8 -*-
# @Author  : cp
# @File    : sample_k_shot.py
import os
import json
import argparse
import random


def update_label_runtime(data, sample_runtime, label_runtime):
    label_sample = [data[sample]['record'] for sample in sample_runtime]

    for label in label_sample:
        for item in label:
            label_runtime[item[-1]] = max(label_runtime[item[-1]] - 1, 0)
    return label_runtime


def sample_data(raw_path, target_path, kshot):

    with open(raw_path, "r", encoding="utf-8") as f:
        movie_train_data = json.loads("[" +
                          f.read().replace("}{", "},\n{") +"]")


    movie_labels = set()  ## 标签集合
    for json_line in movie_train_data:
        for record in json_line['record']:
            if record[-1] != "O":
                movie_labels.add(record[-1])

    label_to_instance = {key: set() for key in movie_labels}

    for idx, json_line in enumerate(movie_train_data):
        records = json_line['record']
        if len(records) >0:
            for record in records:
                label_to_instance[record[-1]].add(idx)

                # if record[0] > 3:## make sure sentence more longer
                #     label_to_instance[record[-1]].add(idx)

    label_length = {key: len(value) for key, value in label_to_instance.items()}  # 排序 先取个数多的实体类型
    label_length = {tup[0]: tup[1] for tup in sorted(label_length.items(), key=lambda x: x[1], reverse=True)}  # 类别: 句子索引数量
    print("label_length;\n",label_length)
    # exit(0)

    sample_train_idx = set()
    label_runtime =  {key:kshot for key in list(label_length.keys())} 
    for key in label_length.keys():
        value = label_to_instance[key] 
        if len(value) < kshot:
            
            sample_train_idx.update(value)
            label_runtime = update_label_runtime(movie_train_data, value, label_runtime)
        else:
            sample_runtime = random.sample(value, k=label_runtime[key])
            label_runtime = update_label_runtime(movie_train_data, sample_runtime, label_runtime)
            sample_train_idx.update(sample_runtime)

    sample_train_data = [movie_train_data[idx] for idx in sample_train_idx]

    with open(os.path.join(target_path,'train.json'), "w", encoding="utf-8", newline='\n') as json_file:
        for data in sample_train_data:
            json.dump(data, json_file, separators=[',', ': '], indent=4, ensure_ascii=False)



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Random seeds")
    parser.add_argument("--data_dir", type=str, default="../data/original/", help="Path to target data")
    parser.add_argument("--data_name", type=str, default="NCBI", help="data name")
    parser.add_argument("--data_file", type=str, default='train.json', choices=['train.json', 'valid.json'], help="k-shot or k-shot-10x (10x valid set)")
    parser.add_argument("--k", type=int, default=10, help="Training examples for each class.")
    parser.add_argument("--mode", type=str, default='k-shot', choices=['k-shot', 'k-shot-10x'], help="k-shot or k-shot-10x (10x valid set)")

    args = parser.parse_args()
    data_path = os.path.join(args.data_dir, args.data_name)
    output_dir = os.path.join(data_path, args.mode)
    raw_dataset = os.path.join(data_path, args.data_file)

    for seed in args.seed:
        random.seed(seed)
        k = args.k
        target_dataset = os.path.join(output_dir, f"{k}-{seed}")
        os.makedirs(target_dataset, exist_ok=True)

        sample_data(raw_dataset, target_dataset, k)




if __name__ == "__main__":
    main()

