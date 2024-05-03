import sys
sys.path.append('.')
import torch
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import argparse, os, json, time, datetime, yaml
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from datasets.sequence_dataset import SequenceDataset, SingleLabelSequenceDataset, MultiLabelSplitDataset, MultiLabelDataset, MLabMergeDataset
from models.mlp import MLPModel
from utils import commons
from utils.losses import NCLoss, pairwise_cosine_distance
from scripts.infer_NC import infer_NC

torch.set_num_threads(4)

def get_args():
    parser = argparse.ArgumentParser(description='Train MLP model')
    parser.add_argument('config', type=str, default='configs/train_mlp.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--logdir', type=str, default='logs_new')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--lambda1', type=float, default=None)
    parser.add_argument('--lambda2', type=float, default=None)
    parser.add_argument('--start_NC_epoch', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--nc1', type=str, default=None)
    parser.add_argument('--no_timestamp', action='store_true')
    parser.add_argument('--random_split_train_val', action='store_true')
    parser.add_argument('--nc_only', action='store_true')
    
    args = parser.parse_args()
    return args

def get_dist_mat(model, means, test_loader, device, log_dir):
    model.eval()
    test_embeddings, test_labels = [], []
    with torch.no_grad():
        for data, label in tqdm(test_loader, desc='Infering train data'):
            data = data.to(device)
            output, features = model(data)
            test_embeddings.append(commons.toCPU(features))
            test_labels.append(label)
        test_embeddings = torch.cat(test_embeddings, dim=0)
        test_labels = torch.cat(test_labels, dim=0)
        
        dist_mat = pairwise_cosine_distance(test_embeddings, means)
    torch.save(commons.toCPU(dist_mat), os.path.join(log_dir, f'dist_mat.pt'))
    
    return dist_mat, test_labels

def get_eval_metrics(predictions, test_labels, logger, log_dir, tag):
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='weighted', zero_division=np.nan)
    recall = recall_score(test_labels, predictions, average='weighted', zero_division=np.nan)
    f1_macro = f1_score(test_labels, predictions, average='macro', zero_division=np.nan)
    f1_micro = f1_score(test_labels, predictions, average='micro', zero_division=np.nan)
    f1_weighted = f1_score(test_labels, predictions, average='weighted', zero_division=np.nan)
    logger.info(f'Accuracy: {accuracy:.4f}')
    logger.info(f'Precision: {precision:.4f}')
    logger.info(f'Recall: {recall:.4f}')
    logger.info(f'F1 macro: {f1_macro:.4f}')
    logger.info(f'F1 micro: {f1_micro:.4f}')
    logger.info(f'F1 weighted: {f1_weighted:.4f}')
    metrics = {'precision': precision, 'recall': recall, 'f1_macro': f1_macro, 'f1_micro': f1_micro, 'f1_weighted': f1_weighted}
    with open(os.path.join(log_dir, f'eval_metrics{tag}.json'), 'w') as f:
        json.dump(metrics, f)
    return metrics

def save_predictions(prediction_mat, test_labels, pids, label_list, save_path, logger):
    n_query, n_classes = prediction_mat.shape
    pred_labels = [[] for _ in range(n_query)]
    gt_labels = [[] for _ in range(n_query)]
    non_zero_indices_pred = torch.nonzero(prediction_mat, as_tuple=False)
    non_zero_indices_gt = torch.nonzero(test_labels, as_tuple=False)
    for index in non_zero_indices_pred:
        row, col = index.tolist()
        pred_labels[row].append(label_list[col])
    for index in non_zero_indices_gt:
        row, col = index.tolist()
        gt_labels[row].append(label_list[col])
    pred_labels = [';'.join(labels) for labels in pred_labels]
    gt_labels = [';'.join(labels) for labels in gt_labels]
    df = pd.DataFrame({'Entry': pids, 'Predictions': pred_labels, 'Ground Truth': gt_labels})
    df.to_csv(save_path, index=False)
    exact_match = 0
    for i in range(n_query):
        if pred_labels[i] == gt_labels[i]:
            exact_match += 1
    exact_match = exact_match / n_query
    logger.info(f'Exact match accuracy: {exact_match:.4f}')

def randomize_ones(tensor):
    # Identify rows with more than one 1
    multiple_ones = tensor.sum(dim=1) > 1
    
    # Iterate over each row with multiple 1s
    for i in torch.where(multiple_ones)[0]:
        # Get indices where the value is 1
        ones_indices = tensor[i].nonzero(as_tuple=True)[0]
        # Randomly choose one index to keep, set others to 0
        keep_index = ones_indices[torch.randint(len(ones_indices), (1,))]
        tensor[i] = 0  # Set all elements to 0
        tensor[i, keep_index] = 1  # Set randomly chosen index to 1

    return tensor

def infer_top1(dist_mat, test_labels, pids, label_list, logger, log_dir, tag):
    # Get the minimum value for each row
    min_vals = dist_mat.min(dim=1, keepdim=True)[0]
    # Compare each element to the minimum value in its row
    predictions = (dist_mat == min_vals).float()
    predictions = randomize_ones(predictions)
    torch.save(predictions, os.path.join(log_dir, f'predictions_top1{tag}.pt'))
    # predictions = predictions.flatten()
    # test_labels = test_labels.flatten()
    # get the accuracy, precision, recall, f1 score
    save_predictions(predictions, test_labels, pids, label_list, os.path.join(log_dir, f'predictions_top1{tag}.csv'), logger)
    metrics = get_eval_metrics(predictions, test_labels, logger, log_dir, tag='_top1' + tag)
    
    return metrics


def main():
    args = get_args()
    # Load configs
    config = commons.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    config.train.seed = args.seed if args.seed is not None else config.train.seed
    commons.seed_all(config.train.seed)
    config.train.lambda1 = args.lambda1 if args.lambda1 is not None or not hasattr(config.train, 'lambda1') else config.train.lambda1
    config.train.lambda2 = args.lambda2 if args.lambda2 is not None or not hasattr(config.train, 'lambda2') else config.train.lambda2
    config.train.start_NC_epoch = args.start_NC_epoch if args.start_NC_epoch is not None or not hasattr(config.train, 'start_NC_epoch') else config.train.start_NC_epoch
    config.data.label_level = 4 if not hasattr(config.data, 'label_level') else config.data.label_level
    config.train.lr = args.lr if args.lr is not None or not hasattr(config.train, 'lr') else config.train.lr
    config.train.weight_decay = args.weight_decay if args.weight_decay is not None or not hasattr(config.train, 'weight_decay') else config.train.weight_decay
    config.train.batch_size = args.batch_size if args.batch_size is not None or not hasattr(config.train, 'batch_size') else config.train.batch_size
    config.train.nc1 = args.nc1 if args.nc1 is not None or not hasattr(config.train, 'nc1') else config.train.nc1

    logger = commons.get_logger('debug')
    log_dir = '.'
    
    config.model.out_dim = 5961
    model = MLPModel(config.model)
    model.to(args.device)
    
    with open(config.data.label_file, 'r') as f:
        label_list = json.load(f)
    
    # Test
    config.train.ckpt_dir = 'logs_ec_new/train_mlp_ec_new_NC_mlab_merge_2024_05_01__01_08_41/checkpoints'
    best_ckpt = torch.load(os.path.join(config.train.ckpt_dir, 'best_checkpoints.pt'))
    model.load_state_dict(best_ckpt)
    best_means = torch.load(os.path.join(config.train.ckpt_dir, 'best_means.pt')).cpu()
    multilabel_testset = MLabMergeDataset(config.data.test_data_file, config.data.label_file, config.data.label_name, config.data.label_level, logger=logger)
    multilabel_test_loader = DataLoader(multilabel_testset, batch_size=config.train.batch_size, shuffle=False)
    pids = multilabel_testset.pids
    logger.info(f'Number of test sequences: {len(multilabel_testset)}')
    
    dist_mat, test_labels = get_dist_mat(model, best_means, multilabel_test_loader, args.device, log_dir)
    test_labels = torch.nn.functional.one_hot(test_labels, num_classes=dist_mat.shape[1]).to(torch.float32)
    logger.info(f'Evaluation on test set using top-1:')
    infer_top1(dist_mat, test_labels, pids, label_list, logger, log_dir, tag='')
    # logger.info(f'Evaluation on test set using max-separation:')
    # infer_maxsep(dist_mat, test_labels, pids, label_list, logger, log_dir, tag='', n=10)
    os.system(f'python scripts/infer_NC_multilabel.py --pred_file {os.path.join(log_dir, "predictions_top1.csv")} --label_file {config.data.original_label_file}')
    
if __name__ == '__main__':
    main()