import os
import numpy as np
import scipy.sparse as ssp
import torch
import torch.nn as nn
import torch.nn.functional as F

utility_path = os.path.split(os.path.realpath(__file__))[0] + "/utility_files/"

class SPROF_GO(nn.Module):
    def __init__(self, task, feature_dim=1024, hidden_dim=256, num_emb_layers=2, num_heads=8, dropout=0.1, device = torch.device('cpu')):
        super(SPROF_GO, self).__init__()

        # Child Matrix: CM_ij = 1 if the jth GO term is a subclass of the ith GO term
        self.CM = torch.tensor(ssp.load_npz(utility_path + task + "_CM.npz").toarray()).to(device)

        # Embedding layers
        self.input_block = nn.Sequential(
                                         nn.LayerNorm(feature_dim, eps=1e-6)
                                        ,nn.Linear(feature_dim, hidden_dim)
                                        ,nn.LeakyReLU()
                                        )

        self.hidden_block = []
        for i in range(num_emb_layers - 1):
            self.hidden_block.extend([
                                      nn.LayerNorm(hidden_dim, eps=1e-6)
                                     ,nn.Dropout(dropout)
                                     ,nn.Linear(hidden_dim, hidden_dim)
                                     ,nn.LeakyReLU()
                                     ])
            if i == num_emb_layers - 2:
                self.hidden_block.extend([nn.LayerNorm(hidden_dim, eps=1e-6)])

        self.hidden_block = nn.Sequential(*self.hidden_block)

        # Self-attention pooling layer
        self.ATFC = nn.Sequential(
                                  nn.Linear(hidden_dim, 64)
                                 ,nn.LeakyReLU()
                                 ,nn.LayerNorm(64, eps=1e-6)
                                 ,nn.Linear(64, num_heads)
                                 )

        # Output layer
        self.label_size = {"MF":790, "BP":4766, "CC":667}[task] # terms with >= 50 samples in the training + validation sets
        self.output_block = nn.Sequential(
                                         nn.Linear(num_heads*hidden_dim, num_heads*hidden_dim)
                                         ,nn.LeakyReLU()
                                         ,nn.LayerNorm(num_heads*hidden_dim, eps=1e-6)
                                         ,nn.Dropout(dropout)
                                         ,nn.Linear(num_heads*hidden_dim, self.label_size)
                                         )

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x, mask, y = None):
        x = self.input_block(x)
        x = self.hidden_block(x)

        # Multi-head self-attention pooling
        att = self.ATFC(x)    # [B, L, num_heads]
        att = att.masked_fill(mask[:, :, None] == 0, -1e9)
        att = F.softmax(att, dim=1)
        att = att.transpose(1,2)   # [B, num_heads, L]
        x = att@x    # [B, num_heads, hidden_dim]
        x = torch.flatten(x, start_dim=1) # [B, num_heads*hidden_dim]

        x = self.output_block(x).sigmoid() # [B, label_size]

        # Hierarchical learning
        if self.training:
            a = (1 - y) * torch.max(x.unsqueeze(1) * self.CM.unsqueeze(0), dim = -1)[0] # the probability of a negative class should take the maximal probabilities of its subclasses
            b = y * torch.max(x.unsqueeze(1) * (self.CM.unsqueeze(0) * y.unsqueeze(1)), dim = -1)[0] # the probability of a positive class should take the maximal probabilities of its positive subclasses
            x = a + b
        else:
            x = torch.max(x.unsqueeze(1) * self.CM.unsqueeze(0), dim = -1)[0]  # [B, 1, label_size] * [1, label_size, label_size]

        return x.float() # [B, label_size]


class SprofECModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.CM = torch.load(config.CM_path).to(self.device)
        self.input_dim = config.input_dim
        self.hidden_dims = config.hidden_dims
        self.out_dim = config.out_dim
        self.dropout = config.dropout
        self.num_layers = len(self.hidden_dims) + 1
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(self.input_dim, self.hidden_dims[0]), 
                                         nn.ReLU(), 
                                         nn.Dropout(self.dropout)))
        for i in range(1, len(self.hidden_dims)):
            self.layers.append(nn.Sequential(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]),
                                             nn.ReLU(),
                                             nn.Dropout(self.dropout)))
        self.layers.append(nn.Linear(self.hidden_dims[-1], self.out_dim))
        
        
        # initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    
    def forward(self, x, y=None):
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
        features = x
        x = self.layers[-1](features)

        # Hierarchical learning
        if self.training:
            a = (1 - y) * torch.max(x.unsqueeze(1) * self.CM.unsqueeze(0), dim = -1)[0] # the probability of a negative class should take the maximal probabilities of its subclasses
            b = y * torch.max(x.unsqueeze(1) * (self.CM.unsqueeze(0) * y.unsqueeze(1)), dim = -1)[0] # the probability of a positive class should take the maximal probabilities of its positive subclasses
            x = a + b
        else:
            x = torch.max(x.unsqueeze(1) * self.CM.unsqueeze(0), dim = -1)[0]  # [B, 1, label_size] * [1, label_size, label_size]

        return x.float(), features # [B, label_size], [B, hidden_dim[-1]]
    
    

if __name__ == '__main__':
    from easydict import EasyDict
    config = EasyDict({'input_dim': 1280, 'hidden_dims': [5120, 2560], 'out_dim': 2205, 'dropout': 0.5, 'CM_path': '../data/ec/CM_EC_all_level_label_above_10.pt', 'device': 'cuda:0'})
    
    model = SprofECModel(config)
    model.to('cuda:0')
    print(model)
    x = torch.randn(8, 1280).to('cuda:0')
    y = torch.randint(0, 2, (8, 2205)).to('cuda:0')
    pred, feat = model(x, y)
    print(pred.shape, feat.shape)
    
    
    