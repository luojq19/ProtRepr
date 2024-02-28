import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import math

class NC1Loss(nn.Module):
    '''
    Modified Center loss, 1 / n_k ||h-miu||
    '''
    def __init__(self, num_classes=10, feat_dim=128, device='cuda:0'):
        super(NC1Loss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        self.means = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.means, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.means.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)
        # print(f'labels: {labels.shape}')
        # input()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        D = torch.sum(dist, dim=0)
        N = mask.float().sum(dim=0) + 1e-10
        # loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        loss = (D / N).clamp(min=1e-12, max=1e+12).sum() / self.num_classes

        return loss, self.means

class NC2Loss(nn.Module):
    '''
    NC2 loss v0: maximize the average minimum angle of each centered class mean
    '''
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, means):
        g_mean = means.mean(dim=0)
        centered_mean = means - g_mean
        means_ = F.normalize(centered_mean, p=2, dim=1)
        cosine = torch.matmul(means_, means_.t())
        # make sure that the diagnonal elements cannot be selected
        cosine = cosine - 2. * torch.diag(torch.diag(cosine))
        max_cosine = cosine.max().clamp(-0.99999, 0.99999)
        # print('min angle:', min_angle)
        # maxmize the minimum angle
        # dim=1 means the maximum angle of the other class to each class
        loss = -torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()
        # loss = cosine.max(dim=1)[0].clamp(-0.99999, 0.99999).mean() + 1. / (means.size(0)-1)

        return loss, max_cosine

class NC2Loss_v1(nn.Module):
    '''
    NC2 loss v1: maximize the minimum angle of centered class means
    '''
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, means):
        g_mean = means.mean(dim=0)
        centered_mean = means - g_mean
        means_ = F.normalize(centered_mean, p=2, dim=1)
        cosine = torch.matmul(means_, means_.t())
        # make sure that the diagnonal elements cannot be selected
        cosine = cosine - 2. * torch.diag(torch.diag(cosine))
        max_cosine = cosine.max().clamp(-0.99999, 0.99999)
        # print('min angle:', min_angle)
        # maxmize the minimum angle
        # dim=1 means the maximum angle of the other class to each class
        loss = -torch.acos(max_cosine)
        min_angle = math.degrees(torch.acos(max_cosine.detach()).item())
        # loss = cosine.max(dim=1)[0].clamp(-0.99999, 0.99999).mean() + 1. / (means.size(0)-1)

        return loss, max_cosine

class NC2Loss_v2(nn.Module):
    '''
    NC2 loss v2: make the cosine of any pair of class-means be close to -1/(C-1))
    '''
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, means):
        C = means.size(0)
        g_mean = means.mean(dim=0)
        centered_mean = means - g_mean
        means_ = F.normalize(centered_mean, p=2, dim=1)
        cosine = torch.matmul(means_, means_.t())
        # make sure that the diagnonal elements cannot be selected
        cosine_ = cosine - 2. * torch.diag(torch.diag(cosine))
        max_cosine = cosine_.max().clamp(-0.99999, 0.99999)
        cosine = cosine_ + (1. - 1/(C-1)) * torch.diag(torch.diag(cosine))
        # print('min angle:', min_angle)
        # maxmize the minimum angle
        loss = cosine.norm()
        # loss = -torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()
        # loss = cosine.max(dim=1)[0].clamp(-0.99999, 0.99999).mean() + 1. / (means.size(0)-1)

        return loss, max_cosine

class NCLoss(nn.Module):
    def __init__(self, sup_criterion, lambda1, lambda2, nc1='NC1Loss', nc2='NC2Loss', num_classes=1920, feat_dim=2000, device='cuda:0'):
        super().__init__()
        self.NC1 = globals()[nc1](num_classes, feat_dim, device)
        self.NC2 = globals()[nc2]()
        self.sup_criterion = globals()[sup_criterion]()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.device = device
        self.num_classes = num_classes
        self.feat_dim = feat_dim
    
    def forward(self, y_pred, labels, features):
        sup_loss = self.sup_criterion(y_pred, labels)
        nc1_loss, means = self.NC1(features, labels)
        nc2_loss, max_cosine = self.NC2(means)
        # print(sup_loss, nc1_loss, nc2_loss)
        # input()
        loss = sup_loss + self.lambda1 * nc1_loss + self.lambda2 * nc2_loss
        return loss, (sup_loss, nc1_loss, nc2_loss, max_cosine)

    def set_lambda(self, lambda1, lambda2):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        print(f'Set weights: lambda1={lambda1}, lambda2={lambda2}')
    
if __name__ == '__main__':
    criterion = NCLoss('CrossEntropyLoss', 0.1, 0.1)
    for param in criterion.parameters():
        print(param.shape)
    y_pred = torch.randn(8, 1920).to('cuda:0')
    labels = torch.randint(0, 1920, (8,)).to('cuda:0')
    features = torch.randn(8, 2000).to('cuda:0')
    print(y_pred.shape, labels.shape, features.shape)
    loss = criterion(y_pred, labels, features)
    print(loss)
