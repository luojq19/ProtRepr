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
from tqdm import tqdm

def get_ec2occurance(data_file, label_file, label_level=4):
    data = torch.load(data_file)
    with open(label_file, 'r') as f:
        label_list = json.load(f)
    ec2occurance = {label: 0 for label in label_list}
    for k, v in data.items():
        for label in v['ec']:
            ec2occurance['.'.join(label.split('.')[:label_level])] += 1
    
    return ec2occurance, label_list

def infer_NC(model, train_loader, test_loader, label_list, device, logger, log_dir, tag):
    tag = f'_{tag}' if tag != '' else ''
    train_embeddings, train_labels = [], []
    with torch.no_grad():
        for data, label in tqdm(train_loader, desc='Infering train data'):
            data = data.to(device)
            output, features = model(data)
            train_embeddings.append(commons.toCPU(features))
            train_labels.extend(label.tolist())
    train_embeddings = torch.cat(train_embeddings, dim=0)
    train_labels = [label_list[i] for i in train_labels]
    label2emb_train = {}
    n = len(train_labels)
    for i in range(n):
        if train_labels[i] not in label2emb_train:
            label2emb_train[train_labels[i]] = []
        label2emb_train[train_labels[i]].append(train_embeddings[i])
    for label in label2emb_train:
        label2emb_train[label] = torch.stack(label2emb_train[label], dim=0).mean(dim=0)
    lookup_labels = list(label2emb_train.keys())
    lookup_embeddings = torch.stack([label2emb_train[label] for label in lookup_labels], dim=0)
    test_embeddings, test_labels, test_outputs = [], [], []
    with torch.no_grad():
        for data, label in tqdm(test_loader, desc='Infering train data'):
            data = data.to(device)
            output, features = model(data)
            test_embeddings.append(commons.toCPU(features))
            test_labels.extend(label.tolist())
            test_outputs.append(commons.toCPU(output))
    test_embeddings = torch.cat(test_embeddings, dim=0)
    test_outputs = torch.cat(test_outputs, dim=0)
    lookup_embeddings_normed = torch.nn.functional.normalize(lookup_embeddings, p=2, dim=1)
    test_embeddings_normed = torch.nn.functional.normalize(test_embeddings, p=2, dim=1)
    # comupte the cosine similarity
    cos_sim = torch.mm(test_embeddings_normed, lookup_embeddings_normed.t())
    # get the top-1 prediction
    preds = cos_sim.argmax(dim=1)
    print(preds)
    print(torch.tensor(test_labels))
    acc = (preds == torch.tensor(test_labels)).float().mean().item()
    logger.info(f'Accuracy: {acc:.4f}')
    output_preds = test_outputs.argmax(dim=1)
    otuput_acc = (output_preds == torch.tensor(test_labels)).float().mean().item()
    logger.info(f'Output accuracy: {otuput_acc:.4f}')
    test_labels = [label_list[i] for i in test_labels]
    test_preds = [lookup_labels[i] for i in preds]
    
    

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
    os.makedirs(os.path.join(config.model_dir, 'eval_NC'), exist_ok=True)
    commons.save_config(config, os.path.join(config.model_dir, 'eval_NC', os.path.basename(args.config)))
    
    train_config = commons.load_config(os.path.join(config.model_dir, 'config.yml'))
    
    # logger
    log_dir = os.path.join(config.model_dir, 'eval_NC')
    logger = commons.get_logger('eval_NC', log_dir)
    logger.info(args)
    logger.info(config)
    
    # dataset
    trainset = SingleLabelSequenceDataset(train_config.data.train_data_file, train_config.data.label_file, train_config.data.label_name, logger=logger)
    testset = SingleLabelSequenceDataset(config.test_data_file, train_config.data.label_file, train_config.data.label_name, logger=logger)
    train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=config.batch_size, shuffle=False)
    train_config.model.out_dim = testset.num_labels
    with open(config.label_file) as f:
        label_list = json.load(f)
    logger.info(f'Loaded {len(label_list)} labels from {config.label_file}')
    
    # model
    model = globals()[train_config.model.model_type](train_config.model)
    ckpt = torch.load(os.path.join(config.model_dir, 'checkpoints', 'best_checkpoints.pt'))
    model.load_state_dict(ckpt)
    logger.info(f'Model loaded from {os.path.join(config.model_dir, "checkpoints", "best_checkpoints.pt")}')
    model.to(args.device)
    model.eval()
    
    # infer with embeddings
    infer_NC(model, train_loader, test_loader, label_list, args.device, logger, log_dir, args.tag)
    
if __name__ == '__main__':
    main()