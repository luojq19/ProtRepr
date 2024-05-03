import sys
sys.path.append('.')

import sys
sys.path.append('.')
import pandas as pd
import os, argparse, json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import commons
from utils.losses import pairwise_cosine_distance
import torch
import numpy as np
from models.mlp import MLPModel
from tqdm import tqdm
from scripts.infer_NC_multilabel import evaluate

def load_model(model_dir, device, logger):
    train_config = commons.load_config(os.path.join(model_dir, 'config.yml'))
    model = MLPModel(train_config.model)
    ckpt = torch.load(os.path.join(model_dir, 'checkpoints', 'best_checkpoints.pt'), map_location='cpu')
    model.load_state_dict(ckpt)
    logger.info(f'Model loaded from {os.path.join(model_dir, "checkpoints", "best_checkpoints.pt")}')
    model.to(device)
    model.eval()
    
    return model

def get_pid_embedding_labels(data_file, model, ont, device, logger):
    data = torch.load(data_file)
    pids, inputs, labels = zip(*[(k, v['embedding'], v[ont]) for k, v in data.items()])
    inputs = torch.vstack(inputs)
    logger.info(f'Loaded {len(inputs)} sequences from {data_file}')
    batch_size = 1000
    num_batches = int(np.ceil(len(inputs) / batch_size))
    embeddings = []
    for i in tqdm(range(num_batches), desc='Embedding', dynamic_ncols=True):
        batch_inputs = inputs[i*batch_size:(i+1)*batch_size].to(device)
        with torch.no_grad():
            batch_embeddings, _ = model(batch_inputs)
        embeddings.append(batch_embeddings.cpu())
    embeddings = torch.cat(embeddings, dim=0).to(device)
    logger.info(f'Embedding shape: {embeddings.shape}')
    
    return pids, embeddings, labels

def infer_lookup_query(lookup_emb, lookup_labels, query_emb, query_labels, query_pids, topk, logger):
    try:
        cos_dist = pairwise_cosine_distance(query_emb, lookup_emb)
    except:
        cos_dist = pairwise_cosine_distance(query_emb.cpu(), lookup_emb.cpu())
    smallest_k_values, smallest_k_indices = commons.n_smallest(cos_dist, topk)
    pred_labels = []
    n_query = len(query_labels)
    for i in tqdm(range(n_query)):
        pred = []
        for j in smallest_k_indices[i]:
            pred.extend(lookup_labels[j])
        pred_labels.append(pred)
    
    pid2pred = {pid: pred for pid, pred in zip(query_pids, pred_labels)}
    pid2gt = {pid: gt for pid, gt in zip(query_pids, query_labels)}
    
    return pid2pred, pid2gt

def save_predictions(pid2pred, pid2gt, save_path):
    pids = list(pid2pred.keys())
    predictions = [';'.join(pid2pred[pid]) for pid in pids]
    ground_truths = [';'.join(pid2gt[pid]) for pid in pids]
    df = pd.DataFrame({'Entry': pids, 'Predictions': predictions, 'Ground Truth': ground_truths})
    df.to_csv(save_path, index=False)

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('config', type=str, default='configs/infer_NC_lookup.yml')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--lookup_data', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--label_file', type=str, default=None)
    parser.add_argument('--topk', type=int, default=None)
    parser.add_argument('--ont', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output', type=str, default='predictions_lookup.csv')
    
    args = parser.parse_args()
    
    return args

def main():
    args = get_args()
    config = commons.load_config(args.config)
    config.model_dir = args.model_dir if args.model_dir is not None else config.model_dir
    config.lookup_data = args.lookup_data if args.lookup_data is not None else config.lookup_data
    config.test_data = args.test_data if args.test_data is not None else config.test_data
    config.label_file = args.label_file if args.label_file is not None else config.label_file
    config.topk = args.topk if args.topk is not None else config.topk
    config.ont = args.ont if args.ont is not None else config.ont
    
    logdir = os.path.join(config.model_dir, 'infer_NC_lookup')
    os.makedirs(logdir, exist_ok=True)
    logger = commons.get_logger('infer', logdir)
    
    model = load_model(config.model_dir, args.device, logger)
    lookup_pids, lookup_emb, lookup_labels = get_pid_embedding_labels(config.lookup_data, model, config.ont, args.device, logger)
    test_pids, test_emb, test_labels = get_pid_embedding_labels(config.test_data, model, config.ont, args.device, logger)
    pid2pred, pid2gt = infer_lookup_query(lookup_emb, lookup_labels, test_emb, test_labels, test_pids, config.topk, logger)
    evaluate(pid2pred, pid2gt, config.label_file, logger)
    
    save_predictions(pid2pred, pid2gt, os.path.join(config.model_dir, args.output))
    
    
if __name__ == '__main__':
    main()


