import sys
sys.path.append('.')
import torch
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import argparse, os, json, time, datetime, yaml
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from datasets.sequence_dataset import SequenceDataset, SingleLabelSequenceDataset
from models.mlp import MLPModel
from utils import commons
from utils.losses import NCLoss

torch.set_num_threads(1)

def get_ec2occurance(data_file, label_file, label_name, label_level):
    data = torch.load(data_file)
    with open(label_file, 'r') as f:
        label_list = json.load(f)
    ec2occurance = {label: 0 for label in label_list}
    for k, v in data.items():
        for label in v[label_name]:
            ec2occurance['.'.join(label.split('.')[:label_level])] += 1
    
    return ec2occurance, label_list

def evaluate(model, val_loader, criterion, device, use_NC=False):
    model.eval()
    all_loss = []
    all_output = []
    if use_NC:
        all_sup_loss, all_nc1_loss, all_nc2_loss, all_max_cosine = [], [], [], []
    with torch.no_grad():
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)
            output, features = model(data)
            # print(output, output.shape)
            # print(label, label.shape)
            # input()
            if use_NC:
                loss, (sup_loss, nc1_loss, nc2_loss, max_cosine) = criterion(output, label, features)
                all_sup_loss.append(commons.toCPU(sup_loss).item())
                all_nc1_loss.append(commons.toCPU(nc1_loss).item())
                all_nc2_loss.append(commons.toCPU(nc2_loss).item())
                all_max_cosine.append(commons.toCPU(max_cosine).item())
            else:
                loss = criterion(output, label)
            all_loss.append(commons.toCPU(loss).item())
            all_output.append(commons.toCPU(output))
        all_loss = torch.tensor(all_loss)
    model.train()
    
    if use_NC:
        return all_loss.mean().item(), (torch.tensor(all_sup_loss).mean().item(), torch.tensor(all_nc1_loss).mean().item(), torch.tensor(all_nc2_loss).mean().item(), torch.tensor(all_max_cosine).mean().item())
    else:
        return all_loss.mean().item()

def train(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, device, logger, config, use_NC=False, writer=None):
    model.train()
    n_bad = 0
    all_loss = []
    all_val_loss = []
    best_val_loss = 1.e10
    epsilon = 1e-4
    if use_NC and config.start_NC_epoch > 0:
        logger.info(f'Not training NC loss until epoch {config.start_NC_epoch}')
        criterion.set_lambda(0, 0)
    for epoch in range(config.num_epochs):
        start_epoch = time.time()
        if use_NC and epoch == config.start_NC_epoch:
            logger.info(f'Start training NC loss at epoch {epoch}')
            criterion.set_lambda(config.lambda1, config.lambda2)
            best_val_loss = 1.e10
        if use_NC:
            val_loss, (val_sup_loss, val_nc1_loss, val_nc2_loss, val_max_cosine) = evaluate(model, val_loader, criterion, device, use_NC)
        else:
            val_loss = evaluate(model, val_loader, criterion, device, use_NC)
        if val_loss > best_val_loss - epsilon:
            n_bad += 1
            if n_bad > config.patience:
                logger.info(f'No performance improvement for {config.patience} epochs. Early stop training!')
                break
        else:
            logger.info(f'New best performance found! val_loss={val_loss:.4f}')
            n_bad = 0
            best_val_loss = val_loss
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(config.ckpt_dir, 'best_checkpoints.pt'))
        all_val_loss.append(val_loss)
        losses = []
        if use_NC:
            sup_losses, nc1_losses, nc2_losses, max_cosines = [], [], [], []
        for data, label in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}', dynamic_ncols=True):
            data = data.to(device)
            label = label.to(device)
            output, features = model(data)
            if use_NC:
                loss, (sup_loss, nc1_loss, nc2_loss, max_cosine) = criterion(output, label, features)
                sup_losses.append(commons.toCPU(sup_loss).item())
                nc1_losses.append(commons.toCPU(nc1_loss).item())
                nc2_losses.append(commons.toCPU(nc2_loss).item())
                max_cosines.append(commons.toCPU(max_cosine).item())
            else:
                loss = criterion(output, label)
            losses.append(commons.toCPU(loss).item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_loss = torch.tensor(losses).mean().item()
        all_loss.append(mean_loss)
        lr_scheduler.step(mean_loss)
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(config.ckpt_dir, 'last_checkpoints.pt'))
        end_epoch = time.time()
        logger.info(f'Epoch [{epoch + 1}/{config.num_epochs}]: loss: {mean_loss:.4f}; val_loss: {val_loss:.4f}; train time: {commons.sec2min_sec(end_epoch - start_epoch)}')
        writer.add_scalar('Train/loss', mean_loss, epoch)
        writer.add_scalar('Val/loss', val_loss, epoch)
        if use_NC:
            writer.add_scalar('Val/sup_loss', val_sup_loss, epoch)
            writer.add_scalar('Val/nc1_loss', val_nc1_loss, epoch)
            writer.add_scalar('Val/nc2_loss', val_nc2_loss, epoch)
            writer.add_scalar('Val/max_cosine', val_max_cosine, epoch)
            writer.add_scalar('Train/sup_loss', torch.tensor(sup_losses).mean().item(), epoch)
            writer.add_scalar('Train/nc1_loss', torch.tensor(nc1_losses).mean().item(), epoch)
            writer.add_scalar('Train/nc2_loss', torch.tensor(nc2_losses).mean().item(), epoch)
            writer.add_scalar('Train/max_cosine', torch.tensor(max_cosines).mean().item(), epoch)
        
    return all_loss, all_val_loss
        
def predict(model, test_loader, device, log_dir, logger):
    model.eval()
    all_output = []
    all_labels = []
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            all_labels.append(label)
            output, _ = model(data)
            all_output.append(output.detach().cpu())
    all_output = torch.cat(all_output, dim=0)
    predictions = torch.argmax(all_output, dim=1).tolist()
    all_labels = torch.cat(all_labels, dim=0).tolist()
    correct = 0
    for i in range(len(predictions)):
        if all_labels[i][predictions[i]] == 1:
            correct += 1
    top1_acc = correct / len(predictions)
    logger.info(f'Top-1 accuracy: {top1_acc:.4f}')
    model.train()
    torch.save(all_output, os.path.join(log_dir, 'logits_test.pt'))
    logger.info(f'Predictions saved to {os.path.join(log_dir, "logits_test.pt")}')

def predict_single_label(model, test_loader, device, log_dir, logger):
    model.eval()
    all_output = []
    all_labels = []
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            all_labels.append(label)
            output, _ = model(data)
            all_output.append(output.detach().cpu())
    all_output = torch.cat(all_output, dim=0)
    predictions = torch.argmax(all_output, dim=1)
    all_labels = torch.cat(all_labels)
    correct = (predictions == all_labels).sum().item()
    top1_acc = correct / len(predictions)
    logger.info(f'Top-1 accuracy: {top1_acc:.4f}')
    model.train()
    torch.save(all_output, os.path.join(log_dir, 'logits_test.pt'))
    logger.info(f'Predictions saved to {os.path.join(log_dir, "logits_test.pt")}')

def multi_level_evaluate(model, test_loader, device, logger, ec2occurance, label_list, log_dir, tag, label_name):
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
    level2acc = {level: level2correct[level] / level2test_num[level] if level2test_num[level] > 0 else 0 for level in occurance_levels}
    for level, acc in level2acc.items():
        logger.info(f'{level}: {acc:.4f}')
    with open(os.path.join(log_dir, f'level2acc_10_30_100{tag}.json'), 'w') as f:
        json.dump(level2acc, f)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), dpi=400)
    axes[0].plot(test_ec2train_occurance_list)
    axes[0].set_xlabel(label_name, fontsize=20)
    axes[0].set_ylabel('Occurance', fontsize=20)
    axes[0].set_title(f'{label_name} occurance in train set', fontsize=20)
    
    axes[1].plot(list(level2acc.values()))
    for i, txt in enumerate(list(level2acc.values())):
        axes[1].annotate(f'{txt:.4f}', (i, list(level2acc.values())[i]), fontsize=10)
    axes[1].set_xticks(range(len(occurance_levels)), occurance_levels)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel(f'{label_name} occurance levels', fontsize=20)
    axes[1].set_ylabel('Accuracy', fontsize=20)
    axes[1].set_title(f'{label_name} accuracy with descending occurance', fontsize=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{label_name}_occurance_acc_10_30_100{tag}.png'), bbox_inches='tight')


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
    
    args = parser.parse_args()
    return args


def main():
    start_overall = time.time()
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

    # Logging
    log_dir = commons.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag, timestamp=True)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    config.train.ckpt_dir = ckpt_dir
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = commons.get_logger('train_mlp', log_dir)
    writer = SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    
    # Load dataset
    trainset = globals()[config.data.dataset_type](config.data.train_data_file, config.data.label_file, config.data.label_name, config.data.label_level, logger=logger)
    validset = globals()[config.data.dataset_type](config.data.valid_data_file, config.data.label_file, config.data.label_name, config.data.label_level, logger=logger)
    testset = globals()[config.data.dataset_type](config.data.test_data_file, config.data.label_file, config.data.label_name, config.data.label_level, logger=logger)
    train_loader = DataLoader(trainset, batch_size=config.train.batch_size, shuffle=True)
    val_loader = DataLoader(validset, batch_size=config.train.batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=config.train.batch_size, shuffle=False)
    config.model.out_dim = trainset.num_labels
    logger.info(f'Trainset size: {len(trainset)}; Validset size: {len(validset)}; Testset size: {len(testset)}')
    logger.info(f'Number of labels: {trainset.num_labels}')
    
    # Load model
    model = MLPModel(config.model)
    model.to(args.device)
    logger.info(model)
    logger.info(f'Trainable parameters: {commons.count_parameters(model)}')
    
    # Train
    if config.train.loss == 'NCLoss':
        logger.info('Using NCLoss')
        criterion = NCLoss(config.train.sup_criterion, config.train.lambda1, config.train.lambda2, num_classes=config.model.out_dim, feat_dim=config.model.hidden_dims[-1], device=args.device, nc1=config.train.nc1, nc2=config.train.nc2)
        optimizer = globals()[config.train.optimizer](list(model.parameters()) + list(criterion.parameters()), lr=config.train.lr, weight_decay=config.train.weight_decay)
    else:
        criterion = globals()[config.train.loss]()
        optimizer = globals()[config.train.optimizer](model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config.train.patience-5, verbose=True)
    train(model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler, device=args.device, logger=logger, config=config.train, use_NC=(config.train.loss == 'NCLoss'), writer=writer)
    
    # Test
    best_ckpt = torch.load(os.path.join(config.train.ckpt_dir, 'best_checkpoints.pt'))
    model.load_state_dict(best_ckpt)
    if config.data.dataset_type == 'SingleLabelSequenceDataset':
        predict_single_label(model=model, test_loader=test_loader, device=args.device, log_dir=log_dir, logger=logger)
    else:
        predict(model=model, test_loader=test_loader, device=args.device, log_dir=log_dir, logger=logger)
    
    commons.save_config(config, os.path.join(log_dir, 'config.yml'))
    
    # multi-level evaluation
    ec2occurance, label_list = get_ec2occurance(config.data.train_data_file, config.data.label_file, label_name=config.data.label_name, label_level=4)
    logger.info(f'Multi-level evaluation on validation set:')
    multi_level_evaluate(model, val_loader, args.device, logger, ec2occurance, label_list, log_dir, tag='val', label_name=config.data.label_name)
    logger.info(f'Multi-level evaluation on test set:')
    multi_level_evaluate(model, test_loader, args.device, logger, ec2occurance, label_list, log_dir, tag='test', label_name=config.data.label_name)
    
    end_overall = time.time()
    logger.info(f'Elapsed time: {commons.sec2min_sec(end_overall - start_overall)}')
    
    
if __name__ == '__main__':
    main()
