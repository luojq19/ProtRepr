import torch
from torch.utils.data import Dataset
import json, logging

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

class SequenceDataset(Dataset):
    def __init__(self, data_file, label_list_file, label_name='ec', label_level=4, logger=None):
        self.logger = logger if logger is not None else get_logger('SequenceDataset')
        self.raw_data = torch.load(data_file)
        self.pids = list(self.raw_data.keys())
        self.embeddings = [self.raw_data[pid]['embedding'] for pid in self.pids]
        self.raw_labels = [self.raw_data[pid][label_name] for pid in self.pids]
        with open(label_list_file, 'r') as f:
            self.label_list = json.load(f)
        self.label2idx = {l: i for i, l in enumerate(self.label_list)}
        self.labels = torch.zeros(len(self.raw_labels), len(self.label_list))
        for i, label in enumerate(self.raw_labels):
            for l in label:
                self.labels[i, self.label2idx['.'.join(l.split('.')[:label_level])]] = 1
        self.logger.info(f'Loaded {len(self)} sequences with {label_name} labels')
        self.num_labels = len(self.label_list)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class SingleLabelSequenceDataset(Dataset):
    def __init__(self, data_file, label_list_file, label_name='ec', label_level=4, logger=None):
        self.logger = logger if logger is not None else get_logger('SequenceDataset')
        self.raw_data = torch.load(data_file)
        self.pids = list(self.raw_data.keys())
        self.embeddings = [self.raw_data[pid]['embedding'] for pid in self.pids]
        self.raw_labels = [self.raw_data[pid][label_name] for pid in self.pids]
        with open(label_list_file, 'r') as f:
            self.label_list = json.load(f)
        self.label2idx = {l: i for i, l in enumerate(self.label_list)}
        self.labels = torch.tensor([self.label2idx['.'.join(l[0].split('.')[:label_level])] for l in self.raw_labels])
        self.logger.info(f'Loaded {len(self)} sequences with {label_name} labels')
        self.logger.info(f'Label level: {label_level}; Num of labels: {len(self.label_list)}')
        self.num_labels = len(self.label_list)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

if __name__ == '__main__':
    dataset = SingleLabelSequenceDataset('../data/ec/sprot_10_1022_esm2_t33_ec_above_10_single_label.pt', '../data/ec/swissprot_ec_list_level3.json', label_name='ec', label_level=3)
    
    emb, label = dataset[0]
    print(emb.shape, label.shape)
    print(label, label.sum())

