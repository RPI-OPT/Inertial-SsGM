# Inertial-SsGM

This repository is the official implementation of the paper: Distributed stochastic inertial methods with delayed derivatives [(Xu et al. 2021)](#xu2021distributed).

<!-- ## Table of Contents
- [Security](#security)
- [Background](#background)
- [Install](#install) 
 -->

## Requirements

Our implementations are in Python 3.8.5 installed under Miniconda. 
Our implementations rely on Pytorch and MPI4PY installed via conda install.

## Content

In the paper, we included 3 examples: phase retrieval problem, neural network models training, and sparse bilinear logistic regression.
Four folders in this repository are corresponding to the three examples (two folders for the neural network models training). 
In each folders, we includes: 
> - **.py, which is the main code for that example;
> - **.sh, which runs the main code with different parameters and stores the results in the subfolder "results";
> - results, which is a folder and incudes the results of the example; 
> - plot_**.py, whihc plots the figures in the paper from the results.
> - **.pdf, which is plotted by plot_**.py 

## Usage

## Performance

On Ubuntu Linux 16.04, Dual Intel Xeon Gold 6130 3.7GHz, 32 CPU cores

## Reference  

- <a name="xu2021distributed"></a>Yangyang Xu, Yibo Xu, Yonggui Yan, and Jie Chen. [Distributed stochastic inertial methods with delayed derivatives](https://arxiv.org/abs/2107.11513). Preprint arXiv:2107.11513, 2021.
