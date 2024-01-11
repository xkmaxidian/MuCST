import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Objective(nn.Module):
    def __init__(self, batch_size, temperature_f=1.):
        super(Objective, self).__init__()
        self.batch_size = batch_size
        self.temperature_f = temperature_f
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def mask_correlated_samples(self, n):
        mask = torch.ones((n, n)).fill_diagonal_(0)
        for i in range(n // 2):
            mask[i, n // 2 + i] = 0
            mask[n // 2 + i, i] = 0
        mask = mask.bool()
        return mask

    # 高级特征对比学习部分的损失计算
    def forward(self, h_i, h_j):
        n = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(-1, 1)
        mask = self.mask_correlated_samples(n)
        negative_samples = sim[mask].reshape(n, -1)

        labels = torch.zeros(n).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss = loss / n
        return loss


def scale_mse(recon_x, x, alpha=0.):
    return F.mse_loss(alpha * (x - recon_x).mean(dim=1, keepdims=True) + recon_x, x)


def contrast_loss(feat1, feat2, tau=0.1, weight=1.):
    sim_matrix = torch.einsum("ik, jk -> ij", feat1, feat2) / torch.einsum(
        "i, j -> ij", feat1.norm(p=2, dim=1), feat2.norm(p=2, dim=1)
    )
    label = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    loss = F.cross_entropy(input=sim_matrix / tau, target=label) * weight
    return loss


def trivial_entropy(feat, tau=0.1, weight=1.):
    prob_x = F.softmax(feat / tau, dim=1)
    p = prob_x / prob_x.norm(p=1).sum(0)
    loss = (p * torch.log(p + 1e-8)).sum() * weight
    return loss


def cross_instance_loss(feat1, feat2, tau=0.1, weight=1.):
    sim_matrix = torch.einsum("ik, jk -> ij", feat1, feat2) / torch.einsum(
        "i, j -> ij", feat1.norm(p=2, dim=1), feat2.norm(p=2, dim=1)
    )
    entropy = torch.distributions.Categorical(logits=sim_matrix / tau).entropy().mean() * weight
    return entropy


def kl_loss(feat_x1, feat_x2, tau):
    return torch.nn.KLDivLoss()(
        (feat_x1 / tau).log_softmax(1), (feat_x2 / tau).softmax(1)
    )
