# hrlcm
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

This is a repo for land cover classification using ensemble labels and high resolution Planet NICFI basemap in Tanzania.

## Introduction

This repo includes all necessary scripts to do land cover mapping in Tanzania using ensemble labels, harmonic fitting coefficients of Sentinel-1 images, and high resolution PlanetScope NICFI basemap. There are three branches:

- main: this is the branch for the standard method taking Northern tanzania as the case study
- no_pad: this is the branch very similar to main branch, except using U-Net without padding
- full_country: this is the branch for mapping the whole Tanzania, but with a simpler method without label quality info.

## Code structure

There are three main parts of scripts for this project:

- data_preprocess: this directory includes all scripts to download and preprocess satellite images.
- guess_model: this directory includes all scripts to build gap-filling Random Forest model and generate ensemble labels.
- hrlcm: this directory includes all scripts to build U-Net model to do land cover mapping.

and other useful scripts and resources:

- tools: other useful scripts for post-mapping processing or use AWS cloud computing.
- docs: useful tutorials or posters/presentations for conference.

## Reference work

Because we have to make a lot of modification, it is hard to fork and change the repo directly. 
So we hard copy the scripts and adapt them. Full credit should be given to the original authors.
All the code within scripts are adapted from:

**Model architecture and workflow**

- [agroimpacts/pytorch_planet](https://github.com/agroimpacts/pytorch_planet), 
- [lukasliebel/dfc2020_baseline](https://github.com/lukasliebel/dfc2020_baseline.git), and
- [schmitt-muc/SEN12MS](https://github.com/schmitt-muc/SEN12MS.git)

**Optimizer**

- [Luolc/AdaBound](https://github.com/Luolc/AdaBound)
- [torch_optimizer](https://pytorch-optimizer.readthedocs.io/en/latest/)

## Acknowledgement

This package is part of project ["Combining Spatially-explicit Simulation of Animal Movement and Earth Observation to Reconcile Agriculture and Wildlife Conservation"](https://github.com/users/LLeiSong/projects/2). This project is funded by NASA FINESST program (award number: 80NSSC20K1640).
