# Detection and Estimation of Structural Consistency (DESC)

## Introduction

DESC is an elegant quadratic programming approach to the problem of group synchronization. 

This repo contains matlab files for implementing the method of the following paper

Robust Group Synchronization via Quadratic Programming, Yunpeng Shi, Cole Wyeth, and Gilad Lerman, ICML 2022

## Usage

Download matlab files. Add paths to the directories Utils, Models, and Algorithms (e.g. addpath Utils). 

Implementations of the algorithms in the paper appear in Algorithms. Results for synthetic data can be reproduced by running Synthetic_Simulations/synthetic_data_experiments.m. Results for real data experiments can be reproduced by loading phototourism data and running the scripts in Real_Data_Experiments, particularly real_DESC and real_MPLS.


Creators:
Yunpeng Shi
shixx517@umn.edu
Cole Wyeth
wyeth008@umn.edu
