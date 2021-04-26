# hrlcm
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

This is a repo for land cover classification using ensemble labels and high resolution Planet NICFI basemap in Tanzania.

## Introduction

## Reference work

Because we have to make a lot of modification, it is hard to fork and change the repo directly. 
So we hard copy the scripts and adapt them. Full credit should be given to the original authors.
All the code within scripts are adapted from:

**Model architecture and workflow**
- [agroimpacts/pytorch_planet](https://github.com/agroimpacts/pytorch_planet), 
- [lukasliebel/dfc2020_baseline](https://github.com/lukasliebel/dfc2020_baseline.git), and
- [schmitt-muc/SEN12MS](https://github.com/schmitt-muc/SEN12MS.git)

**loss functions**
- [bhanML/Co-teaching](https://github.com/bhanML/Co-teaching.git)
- [xingruiyu/coteaching_plus](https://github.com/xingruiyu/coteaching_plus.git)
- [hongxin001/JoCoR](https://github.com/hongxin001/JoCoR.git)

## Acknowledgement

This package is part of project ["Combining Spatially-explicit Simulation of Animal Movement and Earth Observation to Reconcile Agriculture and Wildlife Conservation"](https://github.com/users/LLeiSong/projects/2). This project is funded by NASA FINESST program (award number: 80NSSC20K1640).