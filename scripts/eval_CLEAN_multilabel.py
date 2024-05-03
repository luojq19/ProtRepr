import sys
sys.path.append('.')
import pandas as pd
import os, argparse, json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import commons
import torch
import numpy as np

def read_CLEAN_predictions(pred_file):
    with open(pred_file) as f:
        lines = f.readlines()
    pid2pred = {}
    for line in lines:
        pid, raw_ec = line.strip().split(',', maxsplit=1)
        raw_ec = raw_ec.split(',')
        ecs = [ec.split('/')[0].split(':')[1] for ec in raw_ec]
        pid2pred[pid] = ecs
    
    return pid2pred

def read_ground_truth(data_file):
    data = torch.load(data_file)
    pid2gt = {k: v['ec'] for k, v in data.items()}
    
    return pid2gt

def evaluate(pid2pred, pid2gt, label_file, logger):
    with open(label_file, 'r') as f:
        label_list = json.load(f)
    all_labels = []
    all_preds = []
    for pid, pred in pid2pred.items():
        all_labels.append(pid2gt[pid])
        all_preds.append(pred)
    n_qurey = len(all_labels)
    
    label2idx = {label: idx for idx, label in enumerate(label_list)}
    y_true, y_pred = np.zeros((n_qurey, len(label_list))), np.zeros((n_qurey, len(label_list)))
    for i in range(n_qurey):
        for label in all_labels[i]:
            y_true[i, label2idx[label]] = 1
        for label in all_preds[i]:
            y_pred[i, label2idx[label]] = 1
            
    exact_match = 0
    for i in range(n_qurey):
        if np.all(y_true[i] == y_pred[i]):
            exact_match += 1
    exact_match = exact_match / n_qurey
    
    # y_true = y_true.flatten()
    # y_pred = y_pred.flatten()
    acc = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=np.nan)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=np.nan)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=np.nan)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=np.nan)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=np.nan)
    
    logger.info(f'Accuracy: {acc:.4f}')
    logger.info(f'Precision: {precision:.4f}')
    logger.info(f'Recall: {recall:.4f}')
    logger.info(f'F1_macro: {f1_macro:.4f}')
    logger.info(f'F1_micro: {f1_micro:.4f}')
    logger.info(f'F1_weighted: {f1_weighted:.4f}')
    logger.info(f'Exact match: {exact_match:.4f}')
    
    return acc, precision, recall, f1_macro, f1_micro, f1_weighted, exact_match

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--label_file', type=str, required=True)
    parser.add_argument('--log_dir', type=str, default='logs_CLEAN_new')
    args = parser.parse_args()
    
    return args


def main():
    args = get_args()
    
    logdir = commons.get_new_log_dir(args.log_dir, prefix='eval_CLEAN_multilabel', tag='', timestamp=True)
    logger = commons.get_logger('eval', logdir)

    pid2pred = read_CLEAN_predictions(args.pred_file)
    pid2gt = read_ground_truth(args.test_file)
    
    evaluate(pid2pred, pid2gt, args.label_file, logger)
    
if __name__ == '__main__':
    main()