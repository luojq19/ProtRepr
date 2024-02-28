import sys
sys.path.append('.')
import torch
from torch.utils.data import DataLoader

from utils import commons
from datasets.sequence_dataset import SequenceDataset, SingleLabelSequenceDataset
from models.mlp import MLPModel

import argparse, os, json, time, datetime, yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def get_ec2occurance(data_file, label_file, label_level):
    data = torch.load(data_file)
    with open(label_file, 'r') as f:
        label_list = json.load(f)
    ec2occurance = {label: 0 for label in label_list}
    for k, v in data.items():
        for label in v['ec']:
            ec2occurance['.'.join(label.split('.')[:label_level])] += 1
    
    return ec2occurance, label_list

def evaluate(model, test_loader, device, logger, ec2occurance, label_list, log_dir, tag):
    tag = f'_{tag}' if tag != '' else ''
    model.eval()
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            output, features = model(data)
            all_labels.append(commons.toCPU(label))
            all_outputs.append(commons.toCPU(output))
    all_labels = torch.cat(all_labels, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    preds = all_outputs.argmax(dim=1)
    acc = (preds == all_labels).float().mean().item()
    logger.info(f'Accuracy: {acc:.4f}')
    
    all_labels = all_labels.tolist()
    preds = preds.tolist()
    all_ecs = list(set([label_list[i] for i in all_labels]))
    ec2correct, ec2test_num = {label: 0 for label in all_ecs}, {label: 0 for label in all_ecs}
    test_ec2train_occurance = {ec: ec2occurance[ec] for ec in all_ecs}
    # Sort the dictionary by its values in descending order
    test_ec2train_occurance = dict(sorted(test_ec2train_occurance.items(), key=lambda item: item[1], reverse=True))
    
    n = len(all_labels)
    for i in range(n):
        ec2test_num[label_list[all_labels[i]]] += 1
        if all_labels[i] == preds[i]:
            ec2correct[label_list[all_labels[i]]] += 1
    ec2acc = {ec: ec2correct[ec] / ec2test_num[ec] for ec in all_ecs}
    acc_list_ec_descending = [ec2acc[ec] for ec in test_ec2train_occurance.keys()]
    test_ec2train_occurance_list = list(test_ec2train_occurance.values())
    
    # occurance_levels = ['[500, +$\infty$)', '[100, 500)', '[50, 100)', '[40, 50)', '[30, 40)', '[20, 30)', '[10, 20)', '[5, 10)', '[0, 5)']
    # 0, 10, 30, 100
    occurance_levels = ['[100, +$\infty$)', '[30, 100)', '[10, 30)', '[0, 10)']
    level2correct, level2test_num = {level: 0 for level in occurance_levels}, {level: 0 for level in occurance_levels}
    for ec in all_ecs:
        occurance = test_ec2train_occurance[ec]
        if occurance >= 100:
            level2test_num['[100, +$\infty$)'] += ec2test_num[ec]
            level2correct['[100, +$\infty$)'] += ec2correct[ec]
        elif occurance >= 30:
            level2test_num['[30, 100)'] += ec2test_num[ec]
            level2correct['[30, 100)'] += ec2correct[ec]
        elif occurance >= 10:
            level2test_num['[10, 30)'] += ec2test_num[ec]
            level2correct['[10, 30)'] += ec2correct[ec]
        else:
            level2test_num['[0, 10)'] += ec2test_num[ec]
            level2correct['[0, 10)'] += ec2correct[ec]
        # if occurance >= 500:
        #     level2test_num['[500, +$\infty$)'] += ec2test_num[ec]
        #     level2correct['[500, +$\infty$)'] += ec2correct[ec]
        # elif occurance >= 100:
        #     level2test_num['[100, 500)'] += ec2test_num[ec]
        #     level2correct['[100, 500)'] += ec2correct[ec]
        # elif occurance >= 50:
        #     level2test_num['[50, 100)'] += ec2test_num[ec]
        #     level2correct['[50, 100)'] += ec2correct[ec]
        # elif occurance >= 40:
        #     level2test_num['[40, 50)'] += ec2test_num[ec]
        #     level2correct['[40, 50)'] += ec2correct[ec]
        # elif occurance >= 30:
        #     level2test_num['[30, 40)'] += ec2test_num[ec]
        #     level2correct['[30, 40)'] += ec2correct[ec]
        # elif occurance >= 20:
        #     level2test_num['[20, 30)'] += ec2test_num[ec]
        #     level2correct['[20, 30)'] += ec2correct[ec]
        # elif occurance >= 10:
        #     level2test_num['[10, 20)'] += ec2test_num[ec]
        #     level2correct['[10, 20)'] += ec2correct[ec]
        # elif occurance >= 5:
        #     level2test_num['[5, 10)'] += ec2test_num[ec]
        #     level2correct['[5, 10)'] += ec2correct[ec]
        # else:
        #     level2test_num['[0, 5)'] += ec2test_num[ec]
        #     level2correct['[0, 5)'] += ec2correct[ec]
    level2acc = {level: level2correct[level] / level2test_num[level] if level2test_num[level] > 0 else 0 for level in occurance_levels}
    for level, acc in level2acc.items():
        logger.info(f'{level}: {acc:.4f}')
    with open(os.path.join(log_dir, f'level2acc_10_30_100{tag}.json'), 'w') as f:
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
    plt.savefig(os.path.join(log_dir, f'ec_occurance_acc_10_30_100{tag}.png'), bbox_inches='tight')

def get_args():
    parser = argparse.ArgumentParser(description='Long-Tail Evaluation')
    
    parser.add_argument('config', type=str, default='configs/eval_long_tail.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--test_data_file', type=str, default=None)
    parser.add_argument('--train_data_file', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')
    
    args = parser.parse_args()
    return args

def main():
    start_overall = time.time()
    args = get_args()
    
    # Load configs
    config = commons.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    config.seed = args.seed if args.seed is not None else config.seed
    commons.seed_all(config.seed)
    config.model_dir = args.model_dir if args.model_dir is not None else config.model_dir
    config.test_data_file = args.test_data_file if args.test_data_file is not None else config.test_data_file
    config.train_data_file = args.train_data_file if args.train_data_file is not None else config.train_data_file
    os.makedirs(os.path.join(config.model_dir, 'eval'), exist_ok=True)
    commons.save_config(config, os.path.join(config.model_dir, 'eval', os.path.basename(args.config)))
    
    train_config = commons.load_config(os.path.join(config.model_dir, 'config.yml'))
    
    # logger
    log_dir = os.path.join(config.model_dir, 'eval')
    logger = commons.get_logger('eval', log_dir)
    logger.info(args)
    logger.info(config)
    
    # dataset
    testset = SingleLabelSequenceDataset(config.test_data_file, train_config.data.label_file, train_config.data.label_name, train_config.data.label_level, logger=logger)
    test_loader = DataLoader(testset, batch_size=config.batch_size, shuffle=False)
    train_config.model.out_dim = testset.num_labels
    
    # model
    model = globals()[train_config.model.model_type](train_config.model)
    ckpt = torch.load(os.path.join(config.model_dir, 'checkpoints', 'best_checkpoints.pt'))
    model.load_state_dict(ckpt)
    logger.info(f'Model loaded from {os.path.join(config.model_dir, "checkpoints", "best_checkpoints.pt")}')
    model.to(args.device)
    model.eval()
    
    # evaluation
    ec2occurance, label_list = get_ec2occurance(config.train_data_file, train_config.data.label_file, label_level=train_config.data.label_level)
    evaluate(model, test_loader, args.device, logger, ec2occurance, label_list, log_dir, args.tag)
    

if __name__ == '__main__':
    main()