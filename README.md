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
Each of these four folders includes:
> - **.py, which is the main code for that example;
> - run_**.sh, which runs the main code with different parameters and stores the results in the subfolder "results";
> - results, which is a folder incuding the results of the example; 
> - plot_**.py, whihc plots the figures in the paper from the results;
> - **.pdf, which is plotted by plot_**.py.

Besides the four folders, the repository includes:
> - data, which is a folder including the datasets;
> - run_all.sh， which runs all run_**.sh in the above fours folers.

## Usage

- Run run_all.sh to produce the results in the paper. 
- Or in the example folder, run run_**.sh to produce the results of that example.
- Or run the main code of each example with parameters as the way in run_**.sh.

## Performance

On Ubuntu Linux 16.04, Dual Intel Xeon Gold 6130 3.7GHz, 32 CPU cores. 

![plot](./PhaseRetrieval/PhaseRetrieval_m50000_d20000.pdf?raw=true "title")

![stack Overflow](http://lmsotfy.com/so.png)

## Reference  

- <a name="xu2021distributed"></a>Yangyang Xu, Yibo Xu, Yonggui Yan, and Jie Chen. [Distributed stochastic inertial methods with delayed derivatives](https://arxiv.org/abs/2107.11513). Preprint arXiv:2107.11513, 2021.
