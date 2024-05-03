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
        self.logger.info(f'Loaded {len(self.raw_data)} sequences with {label_name} labels')
        self.logger.info(f'Label level: {label_level}; Num of labels: {len(self.label_list)}')
        self.num_labels = len(self.label_list)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class SingleLabelHierarchyDataset(Dataset):
    def __init__(self, data_file, label_list_file, label_name='ec', label_level=4, logger=None):
        self.logger = logger if logger is not None else get_logger('SequenceDataset')
        self.raw_data = torch.load(data_file)
        self.pids = list(self.raw_data.keys())
        self.embeddings = [self.raw_data[pid]['embedding'] for pid in self.pids]
        self.raw_labels = [self.raw_data[pid][label_name] for pid in self.pids]
        with open(label_list_file, 'r') as f:
            self.label_list = json.load(f)
        self.label2idx = {l: i for i, l in enumerate(self.label_list)}
        # self.labels = torch.tensor([self.label2idx['.'.join(l[0].split('.')[:label_level])] for l in self.raw_labels])
        self.num_labels = len(self.label_list)
        self.labels = torch.zeros(len(self.raw_labels), self.num_labels)
        n = len(self.raw_labels)
        for i in range(n):
            raw_label = self.raw_labels[i]
            hierarchical_labels = ['.'.join(raw_label[0].split('.')[:j]) for j in range(1, 5)]
            for label in hierarchical_labels:
                self.labels[i, self.label2idx[label]] = 1.
        
        self.logger.info(f'Loaded {len(self.raw_data)} sequences with hierarchical {label_name} labels')
        self.logger.info(f'Num of labels: {len(self.label_list)}')

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class MultiLabelSplitDataset(Dataset):
    def __init__(self, data_file, label_list_file, label_name='ec', label_level=4, logger=None) -> None:
        super().__init__()
        self.logger = logger if logger is not None else get_logger('MultilabelDataset')
        self.raw_data = torch.load(data_file)
        self.logger.info(f'Loaded {len(self.raw_data)} raw multi-label sequences')
        with open(label_list_file, 'r') as f:
            self.label_list = json.load(f)
        self.label2idx = {l: i for i, l in enumerate(self.label_list)}
        self.embeddings, self.labels = [], []
        for pid, data in self.raw_data.items():
            for label in data[label_name]:
                self.embeddings.append(data['embedding'])
                label_idx = self.label2idx[label]
                self.labels.append(label_idx)
        self.labels = torch.tensor(self.labels)
        self.logger.info(f'Processed {len(self.embeddings)} single-label sequences')
        self.logger.info(f'Number of labels: {len(self.label_list)}')
        self.num_labels = len(self.label_list)
        
    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]
    
    def __len__(self):
        return len(self.embeddings)

class MultiLabelDataset(Dataset):
    def __init__(self, data_file, label_list_file, label_name='ec', label_level=4, logger=None) -> None:
        super().__init__()
        self.logger = logger if logger is not None else get_logger('MultilabelDataset')
        self.raw_data = torch.load(data_file)
        self.logger.info(f'Loaded {len(self.raw_data)} raw multi-label sequences')
        with open(label_list_file, 'r') as f:
            self.label_list = json.load(f)
        self.label2idx = {l: i for i, l in enumerate(self.label_list)}
        self.embeddings = []
        self.num_data = len(self.raw_data)
        self.labels = torch.zeros(self.num_data, len(self.label_list))
        for i, pid in enumerate(self.raw_data.keys()):
            self.embeddings.append(self.raw_data[pid]['embedding'])
            for label in self.raw_data[pid][label_name]:
                label_idx = self.label2idx[label]
                self.labels[i, label_idx] = 1
        self.logger.info(f'Number of multi-label sequences: {len(self.embeddings)}')
        self.logger.info(f'Number of labels: {len(self.label_list)}')
        self.num_labels = len(self.label_list)
        self.pids = list(self.raw_data.keys())
        
    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]
    
    def __len__(self):
        return len(self.embeddings)
    

class MLabMergeDataset(Dataset):
    '''Merge multilabel to a new single label'''
    def __init__(self, data_file, label_list_file, label_name='ec', label_level=4, logger=None) -> None:
        super().__init__()
        self.logger = logger if logger is not None else get_logger('SequenceDataset')
        self.raw_data = torch.load(data_file)
        self.pids = list(self.raw_data.keys())
        self.embeddings = [self.raw_data[pid]['embedding'] for pid in self.pids]
        self.raw_labels = [self.raw_data[pid][label_name] for pid in self.pids]
        with open(label_list_file, 'r') as f:
            self.label_list = json.load(f)
        self.label2idx = {l: i for i, l in enumerate(self.label_list)}
        self.labels = []
        for i, labels in enumerate(self.raw_labels):
            merged_label = ';'.join(labels)
            self.labels.append(self.label2idx[merged_label])
        self.logger.info(f'Loaded {len(self.raw_data)} sequences with {label_name} labels')
        self.logger.info(f'Number of labels: {len(self.label_list)}')
        self.num_labels = len(self.label_list)
        
    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]
    
    def __len__(self):
        return len(self.embeddings)


if __name__ == '__main__':
    dataset = MLabMergeDataset(data_file='/work/jiaqi/ProtRepr/data/gene3D_new/swissprot_gene3D_by2022-05-25_train.pt', label_list_file='/work/jiaqi/ProtRepr/data/gene3D_new/gene3D_labels_2022-05-25_multilabel_merge.json', label_name='gene3D')
    
    for emb, label in dataset:
        print(emb.shape, label)
        break

