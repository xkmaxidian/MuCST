import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Parameter


class Encoder(nn.Module):
    """
    Encoder for transcript modality
    """

    def __init__(self, in_dims, hidden_dims):
        super().__init__()
        self.in_features = in_dims
        self.hidden_features = hidden_dims

        self.weights = torch.nn.ParameterList()
        self.layers = len(hidden_dims)

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
        return x


class Decoder(nn.Module):
    """
    Decoder for transcript modality
    """

    def __init__(self, hidden_dims, out_dims):
        super().__init__()
        self.hidden_features = hidden_dims
        self.out_features = out_dims

        self.weights = torch.nn.ParameterList()

        current_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:] + [out_dims]:
            weight = Parameter(torch.FloatTensor(current_dim, hidden_dim))
            torch.nn.init.xavier_uniform_(weight)
            self.weights.append(weight)
            current_dim = hidden_dim

    def forward(self, x, adj):
        for i, weight in enumerate(self.weights):
            x = torch.mm(x, weight)
            x = torch.mm(adj, x)
            if i < len(self.weights) - 1:
                x = F.relu(x)
        return x


class Projector(nn.Module):
    """
    Projector part, project features of different modality to shared subspace
    """

    def __init__(self, project_dims):
        super().__init__()

        self.layers = nn.ModuleList()
        in_dim = project_dims[0]

        for out_dim in project_dims[1:]:
            self.layers.append(nn.Sequential(
                nn.Linear(in_features=in_dim, out_features=out_dim),
                nn.ELU(),
                nn.Dropout(0.2),
            ))
            in_dim = out_dim

        if len(project_dims) > 1:
            self.layers.append(nn.Sequential(
                nn.Linear(in_features=in_dim, out_features=project_dims[-1]),
                nn.ELU(),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Discriminator(nn.Module):
    """
    Define the discriminator to distinguish postive sample from negative sample
    """

    def __init__(self, n_h):
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        """
        Initialize the parameters of the discriminator model
        """
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
    """
    We define the readout function here, where the local-avg representation is used to replace the global
    representation in Deep Graph Infomax
    """

    def __init__(self):
        super().__init__()

    def forward(self, emb, mask=None):
        """
        Calculate the avg representation of one micro-environment
        :param emb: the representation of spot
        :param mask: the weighted cell network
        """
        v_sum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((v_sum.shape[1], row_sum.shape[0])).T
        global_emb = v_sum / row_sum
        return F.normalize(global_emb, p=2, dim=1)


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
        self.encoder = ImageEncoder(in_dims=in_features, hidden_dims=out_features)
        self.decoder = ImageDecoder(hidden_dims=out_features, in_dims=in_features)

        self.disc = Discriminator(self.out_features)
        self.sigma = nn.Sigmoid()

    def forward(self, image_feat, aug_image_feat):
        z_raw = self.encoder(image_feat)
        z_aug = self.encoder(aug_image_feat)
        recon_img = self.decoder(z_raw)
        return z_raw, z_aug, recon_img


class MuCST(nn.Module):
    def __init__(self, gene_dims, img_dims, graph_nei, project_dims=[64, 64]):
        super().__init__()
        self.graph = graph_nei
        # in_dim -> hidden -> project[0]
        self.gene_encoder_layer1 = Encoder(in_dims=gene_dims[0], hidden_dims=gene_dims[1:])

        self.decoder = Decoder(hidden_dims=list(reversed(gene_dims[1:])), out_dims=gene_dims[0])

        self.img_encoder = ImageEncoder(in_dims=img_dims[0], hidden_dims=img_dims[-1])
        self.img_decoder = ImageDecoder(hidden_dims=img_dims[-1], in_dims=img_dims[0])

        self.disc = Discriminator(gene_dims[-1])
        self.sigma = nn.Sigmoid()
        self.read = AvgReadout()

        self.projector = Projector(project_dims=project_dims)
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

    def forward(self, gene, fake_gene, img, aug_img1, aug_img2, graph, fusion=False, alpha=1):
        # encode the gene expression into latent embeddings
        zg, hg = self.forward_gene(gene, graph)
        zg_fake, hg_fake = self.forward_gene(fake_gene, graph)

        emb_true = F.relu(zg)
        emb_fake = F.relu(zg_fake)
        g = self.read(emb_true, self.graph)
        g = self.sigma(g)
        g_fake = self.read(emb_fake, self.graph)
        g_fake = self.sigma(g_fake)

        dis_a = self.disc(g, emb_true, emb_fake)

        dis_b = self.disc(g_fake, emb_fake, emb_true)

        zi, hi = self.forward_image(img)
        aug_zi1, aug_hi1 = self.forward_image(aug_img1)
        aug_zi2, aug_hi2 = self.forward_image(aug_img2)

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
        # encode the gene expression into latent embeddings
        zg, hg = self.forward_gene(gene, graph)
        zg_fake, hg_fake = self.forward_gene(fake_gene, graph)

        emb_true = F.relu(zg)
        emb_fake = F.relu(zg_fake)
        g = self.read(emb_true, self.graph)
        g = self.sigma(g)
        g_fake = self.read(emb_fake, self.graph)
        g_fake = self.sigma(g_fake)

        dis_a = self.disc(g, emb_true, emb_fake)

        dis_b = self.disc(g_fake, emb_fake, emb_true)
        zi, hi = self.forward_image(img)

        # reconstruct the gene expression data
        if fusion:
            z_f = alpha * zg + (1 - alpha) * zi
            rec_gene = self.decoder(z_f, graph)
            rec_img = self.img_decoder(z_f)
        else:
            rec_gene = self.decoder(zg, graph)
            rec_img = self.img_decoder(zi)
        return zg, zi, hg, hi, dis_a, dis_b, rec_gene, rec_img
