import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torchvision.models import densenet121, resnet50


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 = sc_1 + s_bias1
        if s_bias2 is not None:
            sc_2 = sc_2 + s_bias2

        logit = torch.cat((sc_1, sc_2), 1)
        return logit


class AvgReadout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, emb, mask=None):
        v_sum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((v_sum.shape[1], row_sum.shape[0])).T
        global_emb = v_sum / row_sum
        return F.normalize(global_emb, p=2, dim=1)


class AutoEncoder(nn.Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu, bottleneck=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act

        self.encoder = Encoder(in_dims=in_features, hidden_dims=out_features)
        self.decoder = Decoder(hidden_dims=out_features, in_dims=in_features)
        self.encoder_bottleneck = Encoder_with_bottleneck(in_dims=in_features, hidden_dims=out_features, bottleneck_dims=32)
        self.decoder_bottleneck = Decoder_with_bottleneck(hidden_dims=out_features, out_dims=in_features, bottleneck_dims=32)

        self.disc = Discriminator(self.out_features)
        self.sigma = nn.Sigmoid()
        self.read = AvgReadout()
        self.bottleneck = bottleneck

    def forward(self, feat, feat_fake, adj):
        if self.bottleneck:
            z = self.encoder_bottleneck.forward(feat, adj)
        else:
            z = self.encoder.forward(feat, adj)
        hidden_emb = z

        # h = torch.mm(z, self.weight2)
        # h = torch.mm(adj, h)
        if self.bottleneck:
            h = self.decoder_bottleneck(z, adj)
        else:
            h = self.decoder(z, adj)
        rec_feature = self.act(h)

        # z_fake = F.dropout(feat_fake, self.dropout, self.training)
        # z_fake = torch.mm(z_fake, self.weight1)
        # z_fake = torch.mm(adj, z_fake)
        if self.bottleneck:
            emb_fake = self.encoder_bottleneck.forward(feat_fake, adj)
        else:
            emb_fake = self.encoder.forward(feat_fake, adj)
        # emb_fake = self.act(z_fake)

        g = self.read(hidden_emb, self.graph_neigh)
        g = self.sigma(g)

        g_fake = self.read(emb_fake, self.graph_neigh)
        g_fake = self.sigma(g_fake)

        ret = self.disc(g, hidden_emb, emb_fake)
        ret_fake = self.disc(g_fake, emb_fake, hidden_emb)
        return hidden_emb, rec_feature, ret, ret_fake


class Encoder(nn.Module):
    def __init__(self, in_dims, hidden_dims):
        super().__init__()
        self.in_features = in_dims
        self.hidden_features = hidden_dims

        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.hidden_features))
        torch.nn.init.xavier_uniform_(self.weight1)

    def forward(self, x, adj):
        z = torch.mm(x, self.weight1)
        z = torch.mm(adj, z)
        # z = F.relu(z)
        return z


class Decoder(nn.Module):
    def __init__(self, hidden_dims, in_dims):
        super().__init__()
        self.hidden_features = hidden_dims
        self.in_features = in_dims

        self.weight = Parameter(torch.FloatTensor(self.hidden_features, self.in_features))
        torch.nn.init.xavier_uniform_(self.weight)
        # self.decoder = nn.Linear(in_features=self.hidden_features, out_features=self.in_features)

    def forward(self, x, adj):
        h = torch.mm(x, self.weight)
        h = torch.mm(adj, h)
        # h = self.decoder(x)
        # return F.relu(h)
        return h


class Encoder_with_bottleneck(nn.Module):
    def __init__(self, in_dims, hidden_dims, bottleneck_dims):
        super().__init__()
        self.in_features = in_dims
        self.hidden_features = hidden_dims

        self.weights = torch.nn.ParameterList()
        self.layers = len(hidden_dims)
        self.bottleneck = nn.Linear(hidden_dims[-1], bottleneck_dims)

        current_dim = in_dims
        for hidden_dim in hidden_dims:
            weight = Parameter(torch.FloatTensor(current_dim, hidden_dim))
            torch.nn.init.xavier_uniform_(weight)
            self.weights.append(weight)
            current_dim = hidden_dim

    def forward(self, x, adj):
        for i in range(self.layers):
            x = torch.mm(x, self.weights[i])
            x = torch.mm(adj, x)
            # in hidden layers, we use ReLU activation
            if i < self.layers - 1:
                x = F.relu(x)
            x = self.bottleneck(x)
        return x


class Decoder_with_bottleneck(nn.Module):
    """
    Decoder for transcript modality
    """

    def __init__(self, hidden_dims, out_dims, bottleneck_dims):
        super().__init__()
        self.hidden_features = hidden_dims
        self.out_features = out_dims

        self.weights = torch.nn.ParameterList()

        current_dim = hidden_dims[0]
        self.bottleneck = nn.Linear(bottleneck_dims, current_dim)
        for hidden_dim in hidden_dims[1:] + [out_dims]:
            weight = Parameter(torch.FloatTensor(current_dim, hidden_dim))
            torch.nn.init.xavier_uniform_(weight)
            self.weights.append(weight)
            current_dim = hidden_dim

    def forward(self, x, adj):
        x = self.bottleneck(x)
        x = F.relu(x)
        for i, weight in enumerate(self.weights):
            x = torch.mm(x, weight)
            x = torch.mm(adj, x)
            if i < len(self.weights) - 1:
                x = F.relu(x)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, in_dims, hidden_dims):
        super().__init__()
        self.in_features = in_dims
        self.hidden_features = hidden_dims
        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_dims, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
        )

    def forward(self, image_feature):
        image_hidden = self.encoder(image_feature)
        return image_hidden


class ImageDecoder(nn.Module):
    def __init__(self, hidden_dims, in_dims):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, in_dims)
        )

    def forward(self, hidden_features):
        reconstruct_image = self.decoder(hidden_features)
        return reconstruct_image


class ImageAutoEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 对应编码器与解码器的参数。我们这里需要尝试，使用图编码器进行编码，再使用简单的全连接层进行解码
        self.encoder = ImageEncoder(in_dims=in_features, hidden_dims=out_features)
        self.decoder = ImageDecoder(hidden_dims=out_features, in_dims=in_features)

        self.disc = Discriminator(self.out_features)
        self.sigma = nn.Sigmoid()

    def forward(self, image_feat, aug_image_feat):
        z_raw = self.encoder(image_feat)
        z_aug = self.encoder(aug_image_feat)
        recon_img = self.decoder(z_raw)
        return z_raw, z_aug, recon_img


class OurModel(nn.Module):
    def __init__(self, gene_dims, img_dims, graph_nei, project_dims=[64, 64]):
        super().__init__()
        self.out_features = gene_dims[1]
        self.graph = graph_nei
        # in_dim -> hidden -> project[0]
        self.gene_encoder_layer1 = Encoder(in_dims=gene_dims[0], hidden_dims=gene_dims[1])
        self.gene_encoder_layer2 = nn.Sequential(
            nn.Linear(in_features=gene_dims[1], out_features=project_dims[0]),
            nn.BatchNorm1d(project_dims[0]),
            nn.ELU(),
            nn.Dropout(0.2)
        )

        self.decoder = Decoder(hidden_dims=project_dims[0], in_dims=gene_dims[0])

        self.img_encoder = ImageEncoder(in_dims=img_dims[0], hidden_dims=project_dims[0])
        self.img_decoder = ImageDecoder(hidden_dims=project_dims[0], in_dims=img_dims[0])

        self.disc = Discriminator(self.out_features)
        self.sigma = nn.Sigmoid()
        self.read = AvgReadout()

        self.projector = nn.Sequential(
            nn.Linear(project_dims[0], project_dims[0]),
            nn.ReLU(),
            nn.Linear(project_dims[0], project_dims[1]),
        )
        self.mse_loss = nn.MSELoss()

    def forward_image(self, image):
        zi = self.img_encoder(image)
        hi = self.projector(zi)
        return zi, hi

    def forward_gene(self, gene, graph):
        zg = self.gene_encoder_layer1.forward(x=gene, adj=graph)
        # z_high_gene = self.gene_encoder_layer2.forward(input=z_low_gene)
        hg = self.projector(zg)
        return zg, hg

    def recon_gene_loss(self, zg, xg, graph):
        zg = self.decoder(zg, graph)
        return self.mse_loss(zg, xg)

    def recon_img_loss(self, zi, xi):
        zi = self.img_decoder(zi)
        return self.mse_loss(zi, xi)

    def forward(self, gene, fake_gene, img, aug_img1, aug_img2, graph, fusion=False, alpha=1):
        # gene编码部分
        zg, hg = self.forward_gene(gene, graph)
        zg_fake, hg_fake = self.forward_gene(fake_gene, graph)

        emb_true = F.relu(zg)
        emb_fake = F.relu(zg_fake)
        g = self.read(emb_true, self.graph)
        g = self.sigma(g)
        g_fake = self.read(emb_fake, self.graph)
        g_fake = self.sigma(g_fake)

        # dis_a: 正样本与局部邻域summary vector的判别性分数
        dis_a = self.disc(g, emb_true, emb_fake)

        # dis_b: 负样本与负样本局部邻域summary vector的判别性分数
        dis_b = self.disc(g_fake, emb_fake, emb_true)

        zi, hi = self.forward_image(img)
        aug_zi1, aug_hi1 = self.forward_image(aug_img1)
        aug_zi2, aug_hi2 = self.forward_image(aug_img2)

        # 重构基因表达: rec_gene
        if fusion:
            z_f = alpha * zg + (1 - alpha) * zi
            rec_gene = self.decoder(z_f, graph)
            rec_img = self.img_decoder(z_f)
        else:
            rec_gene = self.decoder(zg, graph)
            rec_img = self.img_decoder(zi)
        # rec_gene = F.relu(rec_gene)
        return zg, zi, aug_zi1, aug_zi2, hg, hi, aug_hi1, aug_hi2, dis_a, dis_b, rec_gene, rec_img


class OurSimulationModel(nn.Module):
    def __init__(self, gene_dims, img_dims, graph_nei, project_dims=[64, 64]):
        super().__init__()
        self.out_features = gene_dims[1]
        self.graph = graph_nei
        # in_dim -> hidden -> project[0]
        self.gene_encoder_layer1 = Encoder(in_dims=gene_dims[0], hidden_dims=gene_dims[1])
        self.gene_encoder_layer2 = nn.Sequential(
            nn.Linear(in_features=gene_dims[1], out_features=project_dims[0]),
            nn.BatchNorm1d(project_dims[0]),
            nn.ELU(),
            nn.Dropout(0.2)
        )

        self.decoder = Decoder(hidden_dims=project_dims[0], in_dims=gene_dims[0])

        self.img_encoder = ImageEncoder(in_dims=img_dims[0], hidden_dims=project_dims[0])
        self.img_decoder = ImageDecoder(hidden_dims=project_dims[0], in_dims=img_dims[0])

        self.disc = Discriminator(self.out_features)
        self.sigma = nn.Sigmoid()
        self.read = AvgReadout()

        self.projector = nn.Sequential(
            nn.Linear(project_dims[0], project_dims[0]),
            nn.ReLU(),
            nn.Linear(project_dims[0], project_dims[1]),
        )
        self.mse_loss = nn.MSELoss()

    def forward_image(self, image):
        zi = self.img_encoder(image)
        hi = self.projector(zi)
        return zi, hi

    def forward_gene(self, gene, graph):
        zg = self.gene_encoder_layer1.forward(x=gene, adj=graph)
        hg = self.projector(zg)
        return zg, hg

    def recon_gene_loss(self, zg, xg, graph):
        zg = self.decoder(zg, graph)
        return self.mse_loss(zg, xg)

    def recon_img_loss(self, zi, xi):
        zi = self.img_decoder(zi)
        return self.mse_loss(zi, xi)

    def forward(self, gene, fake_gene, img, graph, fusion=False, alpha=1):
        # gene编码部分
        zg, hg = self.forward_gene(gene, graph)
        zg_fake, hg_fake = self.forward_gene(fake_gene, graph)

        emb_true = F.relu(zg)
        emb_fake = F.relu(zg_fake)
        g = self.read(emb_true, self.graph)
        g = self.sigma(g)
        g_fake = self.read(emb_fake, self.graph)
        g_fake = self.sigma(g_fake)

        # dis_a: 正样本与局部邻域summary vector的判别性分数
        dis_a = self.disc(g, emb_true, emb_fake)

        # dis_b: 负样本与负样本局部邻域summary vector的判别性分数
        dis_b = self.disc(g_fake, emb_fake, emb_true)
        zi, hi = self.forward_image(img)

        # 重构基因表达: rec_gene
        if fusion:
            z_f = alpha * zg + (1 - alpha) * zi
            rec_gene = self.decoder(z_f, graph)
            rec_img = self.img_decoder(z_f)
        else:
            rec_gene = self.decoder(zg, graph)
            rec_img = self.img_decoder(zi)
        return zg, zi, hg, hi, dis_a, dis_b, rec_gene, rec_img
