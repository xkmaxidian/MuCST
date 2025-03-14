# MuCST v1.0.6

## MuCST: restoring and integrating heterogeneous morphology images and spatial transcriptomics data with contrastive learning
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10627683.svg)](https://doi.org/10.5281/zenodo.10627683)

###  Yu Wang, Zaiyi Liu, Xiaoke Ma
we present a flexible multi-modal contrastive learning for the integration of spatially resolved transcriptomics (MuCST), including histology image, spatial coordinates and transcription profiles of cells, which jointly perform denoising, elimination of heterogeneity, and compatible feature learning. We demonstrate that MuCST robustly and accurately identifies tissue subpopulations from simulated data with various types of perturbations. In cancer-related tissues, MuCST precisely identifies tumor-associated domains, reveals gene biomarkers for tumor regions, and exposes intratumoral heterogeneity. MuCST is applicable to diverse datasets generated from various platforms, such as STARmap, Visium, and omsFISH for spatial transcriptomics, and hematoxylin and eosin or fluorescence microscopy for images. Overall, MuCST facilitates the integration of multi-modal spatially resolved data, but also serves as pre-processing for data restoration, providing deeper insights into the states, functions, and organization of cells within complex biological tissues.

<img src="docs\MuCST-main.png" alt="\0." style="zoom:24%;" />

# Installation

#### <font color='red'>To accelerate MuCST by using GPU: If you have an NVIDIA GPU, be sure to firstly install a version of PyTorch that supports it (We recommend Pytorch >= 2.0.1). When installing MuCST without install Pytorch previous, the CPU version of torch will be installed by default for you. Here is the [installation guide of PyTorch](https://pytorch.org/get-started/locally/).</font>

#### 1. Start by using python virtual environment with [conda](https://anaconda.org/):

```
conda create --name mucst python=3.9
conda activate mucst
# we recommend install gpu version of torch in previous, use pip will insatll torch-cpu version by default
pip install git+https://github.com/bbchond/torch-toolbox.git@master  # we have fixed bugs in torchtoolbox about numpy
pip install mucstpy
```

(Optional) To run the notebook files in tutorials, please ensure the Jupyter package is installed in your environment:

```
conda install -n mucst ipykernel
python -m ipykernel install --user --name mucst --display-name mucst-jupyter
```

Note: 

1. If you encounter the error message "ImportError: Please install the skmisc package via `pip install --user scikit-misc`" while executing `sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=3000)`, please execute the following command in your terminal: `pip install -i https://test.pypi.org/simple/ scikit-misc==0.2.0rc1`.



## Install R packages

* install R for your device: https://cran.r-project.org/
* Replace the statement `os.environ['R_HOME'] = 'C:/Program Files/R/R-4.3.1'` in Jupyter with your R path
* install.packages("mclust")

## Tutorial

Tutorial notebook for using MuCST integrate morphological features extracted by different models is available at:

https://github.com/xkmaxidian/MuCST/blob/master/tutorials/Histology_Specific_Tutorials.ipynb



The jupyter Notebook of the tutorial for 10 × DLPFC is accessible from:
https://github.com/xkmaxidian/MuCST/blob/master/tutorials/SpatialDomainDLPFC.ipynb

The jupyter notebook of the tutorial for 10 $\times$ Human intestine section A1 is accessible from:

https://github.com/xkmaxidian/MuCST/blob/master/tutorials/SpatialDomainIntestine.ipynb

##### MuCST also applicable to imaging-based ST Platform:

https://github.com/xkmaxidian/MuCST/blob/master/tutorials/SpatialDomainSTARmap.ipynb

##### Note: Full STARmap data are uploaded at our [Zendo](https://zenodo.org/records/10627683).



## System Requirements

#### Python support packages  (Python 3.9.18): 

scanpy, igraph, pandas, numpy, scipy, scanpy, anndata, sklearn, seaborn, torch, tqdm.

For more details of the used package., please refer to 'requirements.txt' file.

##### The coding here is a generalization of the algorithm given in the paper. MuCST is written in Python programming language. To use, please clone this repository and follow the instructions provided in the README.md.



## File Descriptions:

image_feature.py - Extract morphological information from histology image.

model.py - Base code for construct MuCST model.

loss.py - Loss function of MuCST.

utils.py - Auxiliary functions for the MuCST model.

multi_modal_simulation.py - code for simulated multi-modal data generation 

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
* [ConGI](https://github.com/biomed-AI/ConGI)
* [conST](https://github.com/ys-zong/conST)
* [stMVC](https://github.com/cmzuo11/stMVC)
* [stMGATF](https://github.com/liying-1028/stMGATF)
* [Spatial-MGCN](https://github.com/cs-wangbo/Spatial-MGCN/tree/master)

### Contact:

We are continuing adding new features. Bug reports or feature requests are welcome.

Last update: 2025-03-14, version 1.0.7

Please send any questions or found bugs to Xiaoke Ma [xkma@xidian.edu.cn](mailto:xkma@xidian.edu.cn).

### Reference:

Please consider citing the following reference:

- [https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-025-01449-1](https://doi.org/10.1186/s13073-025-01449-1)
