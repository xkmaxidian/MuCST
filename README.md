# MuCST v1.0

## Integrative analysis of spatially resolved transcriptomics data with multi-modal contrastive learning

###  Yu Wang, Xiaoke Ma

we present a flexible multi-modal contrastive learning for the integration of spatially resolved transcriptomics (MuCST), including histology image, spatial coordinates and transcription profiles of cells, which jointly perform denoising, elimination of heterogeneity, and compatible feature learning. We demonstrate that MuCST robustly and accurately identifies tissue subpopulations from simulated data with various types of perturbations. In cancer-related tissues, MuCST precisely identifies tumor-associated domains, reveals gene biomarkers for tumor regions, and exposes intratumoral heterogeneity. MuCST is applicable to diverse datasets generated from various platforms, such as STARmap, Visium, and omsFISH for spatial transcriptomics, and hematoxylin and eosin or fluorescence microscopy for images. Overall, MuCST facilitates the integration of multi-modal spatially resolved data, but also serves as pre-processing for data restoration, providing deeper insights into the states, functions, and organization of cells within complex biological tissues.

![](D:\st_projects\MuCST\docs\MuCST-main.png)

# Installation

please use 'git clone https://github.com/xkmaxidian/MuCST.git'.

## Tutorial

The jupyter Notebook of the tutorial for 10 × DLPFC is accessible from :
https://github.com/xkmaxidian/MuCST/blob/master/tutorials/SpatialDomainDLPFC.ipynb

The jupyter notebook of the tutorial for 10 $\times$ Human intestine section A1 is accessible from:

https://github.com/xkmaxidian/MuCST/blob/master/tutorials/SpatialDomainIntestine.ipynb

##### MuCST also applicable to imaging-based ST Platform:

https://github.com/xkmaxidian/MuCST/blob/master/tutorials/SpatialDomainSTARmap.ipynb

## System Requirements

#### Python support packages  (Python 3.9.18): 

scanpy, igraph, pandas, numpy, scipy, scanpy, anndata, sklearn, seaborn, torch, tqdm.

For more details of the used package., please refer to 'requirements.txt' file.

##### The coding here is a generalization of the algorithm given in the paper. MuCST is written in Python programming language. To use, please clone this repository and follow the instructions provided in the README.md.

## File Descriptions:

image_feature.py: Extract morphological information from histology image.

model.py: Base code for construct MuCST model.

loss.py: Loss function of MuCST.

utils.py - Auxiliary functions for the MuCST model.

## Compared spatial domain identification algorithms

Algorithms that are compared include: 

* [SCANPY](https://github.com/scverse/scanpy-tutorials)
* [Giotto](https://github.com/drieslab/Giotto)
* [stLearn](https://github.com/BiomedicalMachineLearning/stLearn)
* [SEDR](https://github.com/JinmiaoChenLab/SEDR/)
* [BayesSpace](https://github.com/edward130603/BayesSpace)
* [SpaGCN](https://github.com/jianhuupenn/SpaGCN)
* [STAGATE](https://github.com/zhanglabtools/STAGATE)
* [SpatialPCA](https://github.com/shangll123/SpatialPCA)
* [DeepST](https://github.com/JiangBioLab/DeepST)

### Contact:

Please send any questions or found bugs to Xiaoke Ma [xkma@xidian.edu.cn](mailto:xkma@xidian.edu.cn).
