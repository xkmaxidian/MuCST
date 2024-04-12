import os
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import ot
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors, KDTree, BallTree
from torch.backends import cudnn
from tqdm import tqdm
from scipy.sparse import csr_matrix
import sys
from scipy.optimize import linear_sum_assignment


def read_10x_visium(path, count_file='filtered_feature_bc_matrix.h5', library_id=None, load_images=True,
                    quality='hires', image_path=None):
    """
    Method to load 10X Visium dataset
    :param path: the data path
    :param count_file: the name of count file
    :param library_id: the section id of current slice
    :param load_images: whether load the histology image of slice
    :param quality: select which quality of the histology to load
    :param image_path: set the data path of the image
    :return: the loaded AnnData object
    """
    adata = sc.read_visium(path, count_file=count_file, library_id=library_id, load_images=load_images)
    adata.var_names_make_unique()
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
    if quality == "fulres":
        image_coord = adata.obsm["spatial"]
        img = plt.imread(image_path, 0)
        adata.uns["spatial"][library_id]["images"]["fulres"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + quality + "_scalef"]
        image_coord = adata.obsm["spatial"] * scale
    adata.obs["image_col"] = image_coord[:, 0]
    adata.obs["image_row"] = image_coord[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = quality
    return adata


def refine_label(adata, radius=50, key='label'):
    r"""
    Refine the cluster label according to its neighbors
    """
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    return new_type


def cal_spatial_weight(spatial_data, spatial_k, spatial_type='BallTree'):
    r"""
    Construct the cell network
    :param spatial_data: the coordinates of cells, sucha as adata.obsm['spatial']
    :param spatial_k: the number of neighbors of cell
    :param spatial_type: select which way to construct the cell network
    :return: the constructed cell network
    """
    if spatial_type == 'NearestNeighbors':
        nbrs = NearestNeighbors(n_neighbors=spatial_k + 1, algorithm='ball_tree').fit(spatial_data)
        _, indices = nbrs.n_neighbors(spatial_data)
    elif spatial_type == 'KDTree':
        tree = KDTree(spatial_data, leaf_size=2)
        _, indices = tree.query(spatial_data, k=spatial_k + 1)
    elif spatial_type == 'BallTree':
        tree = BallTree(spatial_data, leaf_size=2)
        _, indices = tree.query(spatial_data, k=spatial_k + 1)
    indices = indices[:, 1:]
    spatial_weight = np.zeros((spatial_data.shape[0], spatial_data.shape[0]))
    for i in range(indices.shape[0]):
        ind = indices[i]
        for j in ind:
            spatial_weight[i][j] = 1
    return spatial_weight


def cal_gene_weight(data, n_components=50, gene_dist_type='cosine'):
    r"""
    Calculate the similarity of gene expression between spots
    :param data: the count matrix of gene expression
    :param n_components: reduce the dimension of expression
    :param gene_dist_type: select the similarity type
    :return: the calculated similarity matrix
    """
    if isinstance(data, csr_matrix):
        data = data.toarray()
    if data.shape[1] > 500:
        pca = PCA(n_components=n_components)
        data = pca.fit_transform(data)
    gene_correlation = 1 - pairwise_distances(data, metric=gene_dist_type)
    return gene_correlation


def find_adjacent_spot(adata, use_data='raw', neighbor_k=6):
    r"""
    for cell i, find its neighbors
    """
    if use_data == 'raw':
        if isinstance(adata.X, csr_matrix):
            gene_matrix = adata.X.toarray()
        elif isinstance(adata.X, np.ndarray):
            gene_matrix = adata.X
        elif isinstance(adata.X, pd.DataFrame):
            gene_matrix = adata.X.values
        else:
            raise ValueError(f"{type(adata.X)} is not a valid type.")
    else:
        gene_matrix = adata.obsm[use_data]

    weights_list = []
    final_coordinates = []
    with tqdm(total=len(adata), desc='Find adjacent spots of each spot',
              bar_format='{l_bar}{bar} [Time left: {remaining}]', ) as pbar:
        for i in range(adata.shape[0]):
            current_spot = adata.obsm['weight_phy_mor'][i].argsort()[-neighbor_k:][: -1]
            spot_weight = adata.obsm['weight_phy_mor'][i][current_spot]
            spot_matrix = gene_matrix[current_spot]
            if spot_weight.sum() > 0:
                spot_weight_scaled = (spot_weight / spot_weight.sum())
                weights_list.append(spot_weight_scaled)
                spot_matrix_scaled = np.multiply(spot_weight_scaled.reshape(-1, 1), spot_matrix)
                spot_matrix_final = np.sum(spot_matrix_scaled, axis=0)
            else:
                spot_matrix_final = np.zeros(gene_matrix.shape[1])
                weights_list.append(np.zeros(len(current_spot)))
            final_coordinates.append(spot_matrix_final)
            pbar.update(1)
    adata.obsm['adjacent_data'] = np.array(final_coordinates)
    adata.obsm['adjacent_weight'] = np.array(weights_list)
    return adata


def data_augmentation(adata, k=6, aug_para=0.3):
    if isinstance(adata.X, csr_matrix):
        gene_matrix = adata.X.A
    elif isinstance(adata.X, np.ndarray):
        gene_matrix = adata.X
    elif isinstance(adata.X, pd.DataFrame):
        gene_matrix = adata.X.values
    else:
        raise ValueError(f"{type(adata.X)} is not a valid type here")

    weights_list = []
    final_coordinates = []

    adj_spatial_mor = adata.obsm['mor_adj']
    for i in range(adata.shape[0]):
        current_spot = adj_spatial_mor[i].argsort()[-k:][: -1]
        spot_weight = adj_spatial_mor[i][current_spot]
        spot_feature_matrix = gene_matrix[current_spot]
        if spot_weight.sum() != 0:
            spot_weight_scaled = spot_weight / spot_weight.sum()
            weights_list.append(spot_weight_scaled)
            spot_matrix_scaled = np.multiply(spot_weight_scaled.reshape(-1, 1), spot_feature_matrix)
            spot_matrix_final = np.sum(spot_matrix_scaled, axis=0)
            final_coordinates.append(spot_matrix_final)
        else:
            print(i, current_spot)
            final_coordinates.append(gene_matrix[i])

    adjacent_spot_data = np.array(final_coordinates)

    if isinstance(adata.X, np.ndarray):
        augment_gene_matrix = adata.X + aug_para * adjacent_spot_data.astype(float)
    elif isinstance(adata.X, csr_matrix):
        augment_gene_matrix = adata.X.A + aug_para * adjacent_spot_data.astype(float)
    else:
        raise ValueError(f"{type(adata.X)} is not a valid type here")

    if aug_para > 0:
        adata.obsm['augment_gene_data'] = augment_gene_matrix
    else:
        adata.obsm['augment_gene_data'] = adata.X


def data_augmentation_all(adata, adjacent_weight=0.3, neighbour_k=4, spatial_k=30, n_components=50, md_dist_type='cosine',
                      gb_dist_type='correlation', use_morphological=True, use_data='raw', spatial_type='KDTree'):
    r"""
    Perform data augmentation
    :param adata: input AnnData object
    :param adjacent_weight: the effect of neighbors to current spot, default is 0.3 according to DeepST
    :param neighbour_k: the number of neighbors of spot i in both spatial proximity and morphological similarity
    :param spatial_k: the number of neighbors of spot i in spatial proximity
    :param n_components: the dimension of gene expression to calculate expression similarity
    :param md_dist_type: the metric of similarity
    :param gb_dist_type: the distance metric of gene expression between spots
    :param use_morphological: whether the morphological information is used
    :param use_data: set which feature is employed to find adjacent spots
    :param spatial_type: the method to construct the KNN graph
    :return: the AnnData object, where the augmented data are stored in 'adata.obsm['augment_gene_data']'
    """
    if use_morphological:
        if spatial_type == 'LinearRegress':
            image_row = adata.obs['image_row']
            image_col = adata.obs['image_col']
            array_row = adata.obs['array_row']
            array_col = adata.obs['array_col']
            rate = 3
            reg_row = LinearRegression().fit(array_row.values.reshape(-1, 1), image_row)
            reg_col = LinearRegression().fit(array_col.values.reshape(-1, 1), image_col)
            # calculate the euc distance between spots
            physical_distance = pairwise_distances(adata.obs[['image_col', 'image_row']], metric='euclidean')
            unit = np.sqrt(reg_row.coef_ ** 2 + reg_col.coef_ ** 2)
            physical_distance = np.where(physical_distance >= rate * unit, 0, 1)
        else:
            physical_distance = cal_spatial_weight(adata.obsm['spatial'], spatial_k=spatial_k,
                                                   spatial_type=spatial_type)
    else:
        physical_distance = cal_spatial_weight(adata.obsm['spatial'], spatial_k=spatial_k, spatial_type=spatial_type)

    print('Physical distance calculating done!')
    print('The number of nearest tie neighbors in physical distance is: {}'.format
          (physical_distance.sum() / adata.shape[0]))

    # calculate gene_expression weight
    gene_correlation = cal_gene_weight(adata[:, adata.var['highly_variable']].X.copy(), gene_dist_type=gb_dist_type,
                                       n_components=n_components)
    gene_correlation[gene_correlation < 0] = 0

    print('Gene correlation calculating done!')
    adata.obsm['gene_correlation'] = gene_correlation
    adata.obsm['physical_distance'] = physical_distance

    # calculate image similarity
    if use_morphological:
        morphological_similarity = 1 - pairwise_distances(np.array(adata.obsm['image_feat_pca']), metric=md_dist_type)
        morphological_similarity[morphological_similarity < 0] = 0
        print('Morphological similarity calculating done!')
        adata.obsm['morphological_similarity'] = morphological_similarity
        # MS_{ij} * GC_{ij} * SW_{ij}
        adata.obsm['weight_matrix_all'] = morphological_similarity * gene_correlation * physical_distance
        adata.obsm['weight_phy_mor'] = morphological_similarity * physical_distance
    else:
        # GC_{ij} * SW_{ij}
        adata.obsm['weight_matrix_all'] = gene_correlation * physical_distance
        adata.obsm['weight_phy_mor'] = physical_distance
    print("The weight result of image feature is added to adata.obsm['weights_matrix_all'].")
    adata = find_adjacent_spot(adata=adata, use_data=use_data, neighbor_k=neighbour_k)
    # augment_gene_data
    if isinstance(adata.X, np.ndarray):
        augment_gene_matrix = adata.X + adjacent_weight * adata.obsm['adjacent_data'].astype(float)
    elif isinstance(adata.X, csr_matrix):
        augment_gene_matrix = adata.X.toarray() + adjacent_weight * adata.obsm['adjacent_data'].astype(float)
    adata.obsm['augment_gene_data'] = augment_gene_matrix
    return adata


def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    r"""
    Automatically search for a resolution that matches the target number of clusters
    :param adata: the input AnnData object
    :param n_clusters: the target cluster number
    :param method: which graph cluster method is employed, it should be 'leiden' or 'louvain'
    :param start: the resolution where to start search
    :param end: the resolution where to end
    :param increment: the increment value when the resolution is not right
    :return: the best resolution correspond to the target cluster number
    """
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=20, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, resolution=res, random_state=42)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, resolution=res, random_state=42)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        else:
            raise TypeError(f"Expected `method` to be either `leiden` or `louvain`, found `{str(method)}`.")

        if count_unique == n_clusters:
            label = 1
            break
    assert label == 1, 'Resoultion is not found. Please try bigger range or smaller step!'
    return res


def permutation(feature):
    r"""
    The method to random permute the feature matrix
    :param feature: the raw feature matrix
    :return: the generated feature matrix of the attribute graph
    """
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutate = feature[ids]
    return feature_permutate


def get_feature(adata, use_rep=None):
    r"""
    Store the feature in the 'adata.obsm'
    """
    if use_rep is not None:
        target_feature = adata.obsm[use_rep]
    else:
        target_feature = adata.X

    if isinstance(target_feature, sp.csc_matrix) or isinstance(target_feature, sp.csr_matrix):
        target_feature = target_feature.toarray()

    feature_fake = permutation(target_feature)
    adata.obsm['feat'] = target_feature
    adata.obsm['feat_fake'] = feature_fake


def construction_interaction(adata, n_neighbor=3):
    r"""
    Construct the cell spatial network
    :param adata: the input AnnData object
    :param n_neighbor: the number of neighbors
    """
    position = adata.obsm['spatial']
    distance_matrix = pairwise_distances(position)
    n_spot = adata.shape[0]
    adata.obsm['distance_matrix'] = distance_matrix
    interaction = np.zeros((n_spot, n_spot))
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbor + 1):
            y = distance[t]
            interaction[i, y] = 1
    adata.obsm['graph_neigh'] = interaction
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)
    adata.obsm['adj'] = adj


def add_contrastive_label(adata):
    r"""
    Generate labels for positive samples and negative samples, which is employed to train discriminator
    """
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL


def normalize_adj(adj):
    r"""
    Normalize the adjacency graph
    """
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def preprocess_adj(adj):
    r"""
    Self loop
    """
    adj_normalized = normalize_adj(adj) + np.eye(adj.shape[0])
    return adj_normalized


def fix_seed(seed):
    r"""
    Set seed value, ensure reproduce
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='rec_feat_pca', random_seed=2023):
    r"""
    The cluster method, for Visium dataset, Mclust always perform better than graph-based cluster methods
    :param adata: the input AnnData object
    :param num_cluster: the target cluster number
    :param used_obsm: the latent which is used to input the cluster method
    :param random_seed: seed value
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')


def clustering(adata, n_clusters=7, radius=50, method='mclust', start=0.1, end=3.0, increment=0.01, refinement=False):
    r"""
    The cluster method
    :param adata: the input AnnData object
    :param n_clusters: the target cluster number
    :param radius: the neighbor radius to refine the cluster label
    :param method: the cluster method, should be 'mclust', 'leiden', or 'louvain'
    :param start: the start resolution
    :param end: the end resolution
    :param increment: the increment value for search resolution
    :param refinement: whether to refine the cluster labels
    """
    pca = PCA(n_components=20, random_state=42)
    embedding = pca.fit_transform(adata.obsm['rec_feature'].copy())
    adata.obsm['rec_feat_pca'] = embedding

    if method == 'mclust':
        mclust_R(adata, used_obsm='rec_feat_pca', num_cluster=n_clusters)
        adata.obs['domain'] = adata.obs['mclust']
    elif method == 'leiden':
        res = search_res(adata, n_clusters=n_clusters, use_rep='rec_feat_pca', method=method, start=start, end=end,
                         increment=increment)
        sc.tl.leiden(adata, resolution=res, random_state=42)
        adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
        res = search_res(adata, n_clusters=n_clusters, use_rep='rec_feat_pca', method=method, start=start, end=end,
                         increment=increment)
        sc.tl.louvain(adata, resolution=res, random_state=42)
        adata.obs['domain'] = adata.obs['louvain']

    if refinement:
        new_type = refine_label(adata, radius, key='domain')
        adata.obs['domain'] = new_type


def plot_graph_weights(locations,
                       graph,
                       theta_graph=None,  # azimuthal angles
                       max_weight=1,
                       markersize=1,
                       figsize=(8, 8),
                       title: str = None,
                       flip_yaxis: bool = False,
                       ax=None,
                       ) -> None:
    """
    Visualize weights in a spatial graph,
    heavier weights represented by thicker lines
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    edges, weights, theta = [], [], []

    if theta_graph is not None:
        assert isinstance(theta_graph, csr_matrix)
        assert theta_graph.shape[0] == graph.shape[0]

    for start_node_idx in range(graph.shape[0]):

        ptr_start = graph.indptr[start_node_idx]
        ptr_end = graph.indptr[start_node_idx + 1]

        for ptr in range(ptr_start, ptr_end):
            end_node_idx = graph.indices[ptr]

            # append xs and ys as columns of a numpy array
            edges.append(locations[[start_node_idx, end_node_idx], :])
            weights.append(graph.data[ptr])
            if theta_graph is not None:
                theta.append(theta_graph.data[ptr])

    print(f"Maximum weight: {np.amax(np.array(weights))}\n")
    weights /= np.amax(np.array(weights))

    if theta_graph is not None:
        norm = mpl.colors.Normalize(vmin=min(theta), vmax=max(theta), clip=True)
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap="bwr")
        c = [mapper.to_rgba(t) for t in theta]
    else:
        c = "C0"

    line_segments = LineCollection(
        edges, linewidths=weights * max_weight, linestyle='solid', colors=c, alpha=0.7)
    ax.add_collection(line_segments)

    ax.scatter(locations[:, 0], locations[:, 1], s=markersize, c="gray", alpha=.6, )

    if flip_yaxis:  # 0 on top, typical of image data
        ax.set_ylim(ax.get_ylim()[::-1])

    ax.set_aspect('equal', 'datalim')

    if title is not None:
        ax.set_title(title)


def calculate_overlap(input_data, input_gene_name, used_obs, input_domain_number):
    r"""
    Method to calculate the overlap ratio between marker genes and the identified spatial domain
    :param input_data: the input AnnData object
    :param input_gene_name: the name of makrer gene
    :param used_obs: where the cluster label are stored, such as adata.obs['predict']
    :param input_domain_number: the target domain number, should be inclued in the 'adata.obs[used_obs]'
    :return: the overlap ratio between marker gene and the identified spatial domain
    """
    filtered_adata = input_data[input_data.obs[used_obs] == input_domain_number]
    count_matrix = filtered_adata.X.A
    count_matrix = pd.DataFrame(count_matrix)
    count_matrix.index = filtered_adata.obs.index
    count_matrix.columns = filtered_adata.var.index

    threshold = count_matrix[input_gene_name].min()
    high_expressed_name = input_data[input_data[:, input_gene_name].X > threshold].obs_names
    region_cancer_name = filtered_adata.obs_names

    set_gene_high_expressed = set(high_expressed_name)
    set_cancer_region = set(region_cancer_name)

    overlap = set_gene_high_expressed.intersection(set_cancer_region)
    overlap_ratio = len(overlap) / len(set_cancer_region.union(set_cancer_region))
    return overlap_ratio


def BestMap(L1, L2):
    r"""
    Find the best map for predicted cluster labels to ground truth
    """
    L1 = L1.flatten(order='F').astype(float)
    L2 = L2.flatten(order='F').astype(float)
    if L1.size != L2.size:
        sys.exit('size(L1) must == size(L2)')
    Label1 = np.unique(L1)
    nClass1 = Label1.size
    Label2 = np.unique(L2)
    nClass2 = Label2.size
    nClass = max(nClass1, nClass2)

    # For Hungarian - Label2 are Workers, Label1 are Tasks.
    G = np.zeros([nClass, nClass]).astype(float)
    for i in range(0, nClass2):
        for j in range(0, nClass1):
            G[i, j] = np.sum(np.logical_and(L2 == Label2[i], L1 == Label1[j]))

    _, c = linear_sum_assignment(-G)
    newL2 = np.zeros(L2.shape)
    for i in range(0, nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2
