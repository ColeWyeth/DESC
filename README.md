# Detection and Estimation of Structural Consistency (DESC)

## Introduction

DESC is an elegant quadratic programming approach to the problem of group synchronization. 

This repo contains matlab files for implementing the method of the following paper

[1] Robust Group Synchronization via Quadratic Programming, Yunpeng Shi, Cole Wyeth, and Gilad Lerman, ICML 2022

If you would like to use our code for your paper, please cite [1]. 

## Usage

Download matlab files. Add paths to the directories Utils, Models, and Algorithms (e.g. addpath Utils). 

Implementations of the algorithms in the paper appear in Algorithms. Results for synthetic data can be reproduced by running Synthetic_Simulations/synthetic_data_experiments.m. Results for real data experiments can be reproduced by loading phototourism data and running the scripts in Real_Data_Experiments, particularly real_DESC and real_MPLS.

## Dependencies
The following files in folder ``Utils`` include dependencies for running Lie-Algebraic Averaging method that were written by AVISHEK CHATTERJEE (revised and included in this repo). See also [Robust Rotation Averaging](http://www.ee.iisc.ac.in/labs/cvl/papers/robustrelrotavg.pdf) and [Efficient and Robust Large-Scale Rotation Averaging](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Chatterjee_Efficient_and_Robust_2013_ICCV_paper.pdf) for details.
```
AverageSO3Graph.m
BoxMedianSO3Graph.m
Build_Amatrix.m
ConstantStepSize.m
fmin_adam.m
GCW.m
GlobalSOdCorrectRight.m
HybridGradient.m
IRAAB.m
IterativeSO3Average.m
L12.m
MST.m
PiecewiseStepSize.m
q2R.m
R2q.m
R2Q.m
RobustMeanSO3Graph.m
Rotation_Alignment.m
Weighted_LAA.m
```



Creators:
Yunpeng Shi
yunpengs@princeton.edu
Cole Wyeth
wyeth008@umn.edu
