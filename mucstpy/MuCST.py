import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torchtoolbox.tools import mixup_criterion
from .loss import Objective
from sklearn.decomposition import PCA
from .model import MuCST
from .utils import preprocess_adj
from tqdm import tqdm


def train_model(adata, gene_dims, img_dims, proj_dims=[64, 64], lamb1=0.3, lamb2=0.1, lamb3=1, gamma=1, pre_epoch=1500,
                cont_epoch=50, pc_dims=20, patience=30, adj_key='mor_adj', device='cpu'):
    best_loss = float('inf')
    epochs_since_improvement = 0
    features = torch.FloatTensor(adata.obsm['feat'].copy()).to(device)
    features_fake = torch.FloatTensor(adata.obsm['feat_fake'].copy()).to(device)
    label_CSL = torch.FloatTensor(adata.obsm['label_CSL']).to(device)

    image_feature = torch.FloatTensor(adata.obsm['image_feature']).to(device)
    aug_image_feature1 = torch.FloatTensor(adata.obsm['aug_image_feature1']).to(device)
    aug_image_feature2 = torch.FloatTensor(adata.obsm['aug_image_feature2']).to(device)

    adj = preprocess_adj(adata.obsm[adj_key])
    adj = torch.FloatTensor(adj).to(device)

    model = MuCST(gene_dims=gene_dims, img_dims=img_dims, project_dims=proj_dims, graph_nei=adj).to(device)
    model.train()
    loss_CSL = nn.BCEWithLogitsLoss()
    loss_cont = Objective(batch_size=adata.shape[0])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.)
    print('Begin to train MuCST...')

    if lamb1 == 0:
        total_epoch = pre_epoch
    else:
        total_epoch = pre_epoch + cont_epoch

    epoch_iter = tqdm(range(total_epoch))
    for epoch in epoch_iter:
        latent_gene, latent_img, latent_aug_img1, latent_aug_img2, high_gene, high_img, high_aug_img1, high_aug_img2, ret, ret_fake, rec_data, rec_img = model.forward(
            features, features_fake, image_feature, aug_image_feature1, aug_image_feature2, adj)
        loss_sl_1 = loss_CSL(ret, label_CSL)
        loss_sl_2 = loss_CSL(ret_fake, label_CSL)
        loss_g2g = loss_sl_1 + loss_sl_2

        loss_feat = F.mse_loss(features, rec_data)
        loss_img = F.mse_loss(image_feature, rec_img)

        loss_recon = loss_feat + lamb1 * loss_img
        if epoch < pre_epoch:
            loss = loss_recon + lamb2 * loss_g2g
            # pre_train_loss.append(loss.item())
        else:
            loss_i2i = loss_cont(high_aug_img1, high_aug_img2)
            lam = np.random.beta(0.2, 0.2)
            loss_i2g = mixup_criterion(loss_cont, high_gene, high_img, high_aug_img1, lam)
            loss = loss_recon + lamb2 * loss_g2g + gamma * loss_i2i + lamb3 * loss_i2g
            # cont_loss.append(loss.item())

        if epoch == pre_epoch:
            best_loss = float("inf")
            print(f"# Epoch {epoch}, loss: {loss.item():.3f}, g2g_loss: {loss_g2g.item():.3f}, gene_recon: {loss_feat.item():.3f}, image_recon: {loss_img.item():.3f}")
            print('Init finished, start contrastive learning part')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            epochs_since_improvement = 0
        else:
            epochs_since_improvement = epochs_since_improvement + 1

        if epochs_since_improvement == patience:
            print(f"Training stopped after {epoch + 1} epochs due to no improvement in loss.")
            break

        if epoch < pre_epoch:
            epoch_iter.set_description(
                f"# Epoch {epoch}, loss: {loss.item():.3f}, g2g_loss: {loss_g2g.item():.3f}, gene_recon: {loss_feat.item():.3f}, image_recon: {loss_img.item():.3f}")
        else:
            epoch_iter.set_description(
                f"# Epoch {epoch}, loss: {loss.item():.3f}, g2i_loss: {loss_i2g.item():.3f}, g2g_loss: {loss_g2g.item():.3f}, i2i_loss: {loss_i2i.item():.3f}, gene_recon: {loss_feat.item():.3f}, image_recon: {loss_img.item():.3f}")
    print('Optimization of MuCST finished')

    with torch.no_grad():
        model.eval()
        latent_gene, latent_img, latent_aug_img1, latent_aug_img2, high_gene, high_img, high_aug_img1, high_aug_img2, ret, ret_fake, rec_data, rec_img = model.forward(
            features, features_fake, image_feature, aug_image_feature1, aug_image_feature1, adj)
        latent_gene = latent_gene.detach().cpu().numpy()
        latent_img = latent_img.detach().cpu().numpy()
        rec_data = rec_data.detach().cpu().numpy()
        rec_img = rec_img.detach().cpu().numpy()

    adata.obsm['rec_feature'] = rec_data
    adata.layers['My_ReX'] = rec_data

    alpha = lamb1
    pca = PCA(n_components=pc_dims, random_state=2023)
    rec_feat_pca = pca.fit_transform(rec_data)
    rec_img_pca = pca.fit_transform(rec_img)
    adata.obsm['rec_feat_pca'] = rec_feat_pca
    adata.obsm['rec_img_pca'] = rec_img_pca
    adata.obsm['fusion_pca'] = rec_feat_pca + alpha * rec_img_pca
    adata.obsm['latent_gene'] = latent_gene
    adata.obsm['latent_img'] = latent_img
    # return adata
    torch.cuda.empty_cache()