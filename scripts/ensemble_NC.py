import sys
sys.path.append('.')
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils import commons
from datasets.sequence_dataset import SequenceDataset, SingleLabelSequenceDataset
from models.mlp import MLPModel
from utils.losses import pairwise_cosine_distance
from scripts.infer_NC import get_ec2occurance

import argparse, os, json, time, datetime, yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter

def load_model(model_dir, device, logger):
    train_config = commons.load_config(os.path.join(model_dir, 'config.yml'))
    model = globals()[train_config.model.model_type](train_config.model)
    ckpt = torch.load(os.path.join(model_dir, 'checkpoints', 'best_checkpoints.pt'))
    model.load_state_dict(ckpt)
    logger.info(f'Model loaded from {os.path.join(model_dir, "checkpoints", "best_checkpoints.pt")}')
    model.to(device)
    model.eval()
    learned_means = torch.load(os.path.join(model_dir, 'checkpoints/means.pt')).to(device)
    
    return model, learned_means

def infer_NC(model, learned_means, test_loader, device, tag):
    test_embeddings, test_labels, test_outputs = [], [], []
    with torch.no_grad():
        for data, label in tqdm(test_loader, desc=f'Model {tag}'):
            data = data.to(device)
            output, features = model(data)
            test_embeddings.append(features)
            test_labels.extend(label.tolist())
            test_outputs.append(commons.toCPU(output))
    test_embeddings = torch.cat(test_embeddings, dim=0)
    test_outputs = torch.cat(test_outputs, dim=0)
    cos_dist = pairwise_cosine_distance(test_embeddings, learned_means)
    preds = cos_dist.argmin(dim=1)
    
    return commons.toCPU(preds), commons.toCPU(cos_dist)

def ensemble_tensors_by_voting(tensor_list):
    """
    Ensemble tensors by voting for each element.
    
    Parameters:
    tensors (torch.Tensor): A list of 1-D torch tensors of the same length.

    Returns:
    torch.Tensor: A 1-D tensor representing the ensemble of the input tensors by voting.
    """
    # Stack the tensors to create a 2D tensor where each row represents an original tensor
    stacked_tensors = torch.stack(tensor_list)
    
    # The resulting tensor
    result = torch.empty_like(stacked_tensors[0])
    
    # Iterate through each element position
    for i in range(stacked_tensors.shape[1]):
        # Get all elements at the i-th position across all tensors
        elements = stacked_tensors[:, i]
        
        # Count the occurrences of each element and find the most common one
        most_common_element, _ = Counter(elements.tolist()).most_common(1)[0]
        
        # Assign the most common element to the result tensor
        result[i] = most_common_element
    
    return result

def get_multi_level_acc(y_pred, y_true, ec2occurance, label_list, logger, log_dir, levels=[10, 30, 100]):
    acc = (y_pred == y_true).float().mean().item()
    logger.info(f'Accuracy: {acc:.4f}')
    pred_labels = [label_list[i] for i in y_pred]
    gt_labels = [label_list[i] for i in y_true]
    level2acc = commons.get_leveled_acc(pred_labels, gt_labels, ec2occurance, log_dir, levels=levels)
    for level in level2acc:
        logger.info(f'{level}: {level2acc[level]:.4f}')
    with open(os.path.join(log_dir, f'level2acc.json'), 'w') as f:
        json.dump(level2acc, f)

def get_args():
    parser = argparse.ArgumentParser(description='Ensemble NC')
    
    parser.add_argument('config', type=str, default='configs/eval_long_tail.yml')
    parser.add_argument('--logdir', type=str, default='logs_ensemble')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--no-timestamp', action='store_true')
    
    args = parser.parse_args()
    return args

def main():
    start_overall = time.time()
    args = get_args()
    
    # Load configs
    config = commons.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    
    # Logging
    log_dir = commons.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag, timestamp=not args.no_timestamp)
    logger = commons.get_logger('ensemble_infer', log_dir)
    logger.info(args)
    logger.info(config)
    commons.save_config(config, os.path.join(log_dir, os.path.basename(args.config)))
    
    # dataset
    # trainset = SingleLabelSequenceDataset(config.train_data_file, config.label_file, config.label_name, logger=logger)
    testset = SingleLabelSequenceDataset(config.test_data_file, config.label_file, config.label_name, logger=logger)
    test_loader = DataLoader(testset, batch_size=config.batch_size, shuffle=False)
    with open(config.label_file) as f:
        label_list = json.load(f)
        
    # models
    model_list, learned_means_list = [], []
    for model_dir in config.model_dir_list:
        model, learned_means = load_model(model_dir, args.device, logger)
        model_list.append(model)
        learned_means_list.append(learned_means)
    n = len(model_list)
    logger.info(f'Ensmeble {n} models')
    
    # Infer
    ec2occurance, label_list = get_ec2occurance(config.train_data_file, config.label_file, label_level=config.label_level)
    
    pred_list = []
    dist_list = []
    for i in range(n):
        pred, dist = infer_NC(model_list[i], learned_means_list[i], test_loader, args.device, i)
        pred_list.append(pred)
        dist_list.append(dist)
    # ensembled_pred = ensemble_tensors_by_voting(pred_list)
    ensembled_dist = sum(dist_list) / n
    ensembled_pred = ensembled_dist.argmin(dim=1)
    print(ensembled_pred, ensembled_pred.shape)
    
    
    test_labels =testset.labels
    get_multi_level_acc(ensembled_pred, test_labels, ec2occurance, label_list, logger, log_dir)
    
if __name__ == '__main__':
    main()