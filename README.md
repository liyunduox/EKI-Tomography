# EKI-Tomography
 This repository contains the code for the article: "Seismic traveltime tomography based on ensemble Kalman inversion" Published in Geophysical Journal International (DOI: http://dx.doi.org/10.1093/gji/ggab287)
# Hardware Requirements
CPU 4-core processor (Recommended 10-core processor or higher)

RAM 16GB (Recommended 64GB or higher)
# Operating System	Requirements
Windows
# Software Requirements
Julia	1.8.1
# Package Requirements
Distributions, StatsBase, Optim, QuadGK, Plots, PyPlot, JLD2, MAT, FLUX, jInv, DocStringExtensions, GaussianRandomFields
# Already integrated Packages
EnsembleKalmanProcesses (https://github.com/CliMA/EnsembleKalmanProcesses.jl)

FactoredEikonalFastMarching (https://github.com/JuliaInv/FactoredEikonalFastMarching.jl)
# Citing
If you find this package useful in your work, feel free to cite
```bash
@article{10.1093/gji/ggae329,
    author = {Li, Yunduo and Zhang, Yijie and Zhu, Xueyu and Gao, Jinghuai},
    title = {Seismic traveltime tomography based on ensemble Kalman inversion},
    journal = {Geophysical Journal International},
    volume = {240},
    number = {1},
    pages = {290-302},
    year = {2024},
    month = {09},
    issn = {1365-246X},
    doi = {10.1093/gji/ggae329},
    url = {https://doi.org/10.1093/gji/ggae329},
    eprint = {https://academic.oup.com/gji/article-pdf/240/1/290/60814646/ggae329.pdf},
}

