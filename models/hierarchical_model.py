import torch
import torch.nn as nn


class HierarchicalMLPModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.input_dim = config.input_dim
        self.hidden_dims = config.hidden_dims
        self.embedding_dims = config.embedding_dims
        self.out_dims = config.out_dims
        self.num_hierarchy = len(self.out_dims)
        assert len(self.out_dims) == len(self.embedding_dims)
        self.dropout = config.dropout
        self.num_layers = len(self.hidden_dims) + 1
        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Sequential(nn.Linear(self.input_dim, self.hidden_dims[0]), 
                                      nn.ReLU(), 
                                      nn.Dropout(self.dropout)))
        for i in range(1, len(self.hidden_dims)):
            self.mlp.append(nn.Sequential(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]),
                                          nn.ReLU(),
                                          nn.Dropout(self.dropout)))
        self.embedders = nn.ModuleList()
        self.predictors = nn.ModuleList()
        self.embedders.append(nn.Sequential(nn.Linear(self.hidden_dims[-1], self.embedding_dims[0]),
                                            nn.ReLU(),
                                            nn.Dropout(self.dropout)))
        for i in range(1, self.num_hierarchy):
            self.embedders.append(nn.Sequential(nn.Linear(self.embedding_dims[i-1], self.embedding_dims[i]),
                                                nn.ReLU(),
                                                nn.Dropout(self.dropout)))
        for i in range(self.num_hierarchy):
            self.predictors.append(nn.Sequential(nn.Linear(self.embedding_dims[i], self.out_dims[i]),
                                                 nn.ReLU(),
                                                 nn.Dropout(self.dropout)))
        
        
    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.mlp[i](x)
        embeddings, predictions = [], []
        for i in range(self.num_hierarchy):
            x = self.embedders[i](x)
            pred = self.predictors[i](x)
            embeddings.append(x)
            predictions.append(pred)
        
        return predictions, embeddings
    
if __name__ == '__main__':
    from easydict import EasyDict
    config = EasyDict({'input_dim': 1280, 'hidden_dims': [2560, 2000], 'embedding_dims': [2000, 500, 100, 20], 'out_dims': [1920, 211, 67, 7], 'dropout': 0.5})
    
    model = HierarchicalMLPModel(config)
    model.to('cuda:0')
    print(model)
    x = torch.randn(8, 1280).to('cuda:0')
    predictions, embeddings = model(x)
    for pred in predictions:
        print(pred.shape)
    for embed in embeddings:
        print(embed.shape)