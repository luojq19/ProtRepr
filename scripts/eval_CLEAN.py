import sys
sys.path.append('.')
import torch
import pandas as pd
import argparse, os, json, time, datetime, yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from utils import commons

def get_ec2occurance(data_file, label_file, label_level):
    data = torch.load(data_file)
    with open(label_file, 'r') as f:
        label_list = json.load(f)
    ec2occurance = {label: 0 for label in label_list}
    for k, v in data.items():
        for label in v['ec']:
            ec2occurance['.'.join(label.split('.')[:label_level])] += 1
    
    return ec2occurance, label_list

def read_CLEAN_predictions(pred_file):
    with open(pred_file) as f:
        lines = f.readlines()
    pids = [line.split(',')[0] for line in lines]
    ecs = [line.split(',')[-1].strip() for line in lines]
    top1_ecs = [ec.split('/')[0].split(':')[1] for ec in ecs]
    pid2pred = {pid: ec for pid, ec in zip(pids, top1_ecs)}
    
    return pid2pred

def read_ground_truth(data_file):
    data = torch.load(data_file)
    pid2gt = {k: v['ec'][0] for k, v in data.items()}
    
    return pid2gt

def evaluate(pid2pred, pid2gt, ec2occurance, label_list, logger, log_dir):
    all_labels = list(pid2gt.values())
    preds = [pid2pred[pid] for pid in pid2gt]
    all_ecs = list(set(all_labels))
    ec2correct, ec2test_num = {label: 0 for label in all_ecs}, {label: 0 for label in all_ecs}
    test_ec2train_occurance = {ec: ec2occurance[ec] for ec in all_ecs}
    # Sort the dictionary by its values in descending order
    test_ec2train_occurance = dict(sorted(test_ec2train_occurance.items(), key=lambda item: item[1], reverse=True))
    
    n = len(all_labels)
    correct = 0
    for i in range(n):
        if all_labels[i] == preds[i]:
            correct += 1
    logger.info(f'Accuracy: {correct / n:.4f}')
    for i in range(n):
        ec2test_num[all_labels[i]] += 1
        if all_labels[i] == preds[i]:
            ec2correct[all_labels[i]] += 1
    ec2acc = {ec: ec2correct[ec] / ec2test_num[ec] for ec in all_ecs}
    acc_list_ec_descending = [ec2acc[ec] for ec in test_ec2train_occurance.keys()]
    test_ec2train_occurance_list = list(test_ec2train_occurance.values())
    occurance_levels = ['[100, +$\infty$)', '[30, 100)', '[0, 30)']
    level2correct, level2test_num = {level: 0 for level in occurance_levels}, {level: 0 for level in occurance_levels}
    for ec in all_ecs:
        occurance = test_ec2train_occurance[ec]
        if occurance >= 100:
            level2test_num['[100, +$\infty$)'] += ec2test_num[ec]
            level2correct['[100, +$\infty$)'] += ec2correct[ec]
        elif occurance >= 30:
            level2test_num['[30, 100)'] += ec2test_num[ec]
            level2correct['[30, 100)'] += ec2correct[ec]
        # elif occurance >= 10:
        #     level2test_num['[10, 30)'] += ec2test_num[ec]
        #     level2correct['[10, 30)'] += ec2correct[ec]
        else:
            level2test_num['[0, 30)'] += ec2test_num[ec]
            level2correct['[0, 30)'] += ec2correct[ec]
    level2acc = {level: level2correct[level] / level2test_num[level] if level2test_num[level] > 0 else 0 for level in occurance_levels}
    for k, v in level2acc.items():
        logger.info(f'{k}: {v:.4f};')
    logger.info('')
    with open(os.path.join(log_dir, 'level2acc_10_30_100.json'), 'w') as f:
        json.dump(level2acc, f)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=400)
    axes[0].plot(test_ec2train_occurance_list)
    axes[0].set_xlabel('ECs', fontsize=20)
    axes[0].set_ylabel('Occurance', fontsize=20)
    axes[0].set_title('EC Occurance in train set', fontsize=20)
    
    axes[1].plot(list(level2acc.values()))
    for i, txt in enumerate(list(level2acc.values())):
        axes[1].annotate(f'{txt:.4f}', (i, list(level2acc.values())[i]), fontsize=10)
    axes[1].set_xticks(range(len(occurance_levels)), occurance_levels)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel('EC occurance levels', fontsize=20)
    axes[1].set_ylabel('Accuracy', fontsize=20)
    axes[1].set_title('EC Accuracy with descending occurance', fontsize=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'ec_occurance_acc_10_30_100.png'), bbox_inches='tight')
    
def get_args():
    parser = argparse.ArgumentParser(description='Long-Tail Evaluation')
    
    parser.add_argument('config', type=str, default='configs/eval_long_tail.yml')
    parser.add_argument('--logdir', type=str, default='logs_clean')
    parser.add_argument('--clean_pred_file', type=str, required=True)
    parser.add_argument('--test_data_file', type=str, default=None)
    parser.add_argument('--train_data_file', type=str, default=None)
    parser.add_argument('--label_file', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')
    
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    config = commons.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    config.test_data_file = args.test_data_file if args.test_data_file is not None else config.test_data_file
    config.train_data_file = args.train_data_file if args.train_data_file is not None else config.train_data_file
    config.label_file = args.label_file if args.label_file is not None else config.label_file
    
    # logger
    log_dir = commons.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag, timestamp=True)
    logger = commons.get_logger('eval', log_dir)
    logger.info(args)
    logger.info(config)
    commons.save_config(config, os.path.join(log_dir, 'config.yml'))
    
    # read CLEAN predictions
    pid2pred = read_CLEAN_predictions(args.clean_pred_file)
    pid2gt = read_ground_truth(config.test_data_file)
    
    # evaluation
    ec2occurance, label_list = get_ec2occurance(config.train_data_file, config.label_file, label_level=4)
    evaluate(pid2pred, pid2gt, ec2occurance, label_list, logger, log_dir)
    
if __name__ == '__main__':
    main()
