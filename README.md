# CCN
## Connective Cognition Network for Directional Visual Commonsense Reasoning (NeurIPS 2019)

![Method](https://github.com/AmingWu/CCN/blob/master/pic/fig1.png?raw=true "Illustration of our method")
Visual commonsense reasoning (VCR) has been introduced to boost research of cognition-level visual understanding, i.e., a thorough understanding of correlated details of the scene plus an inference with related commonsense knowledge. We propose a connective cognition network (CCN) to dynamically reorganize the visual neuron connectivity that is contextualized by the meaning of questions and answers. And our method mainly includes visual neuron connectivity, contextualized connectivity, and directional connectivity.

![Framework](https://github.com/AmingWu/CCN/blob/master/pic/fig2.png?raw=true "Illustration of our framework")

The goal of visual neuron connectivity is to obtain a global representation of an image, which is helpful for a thorough understanding of visual content. It mainly includes visual element connectivity and the computation of both conditional centers and GraphVLAD.

![Visual Neuron Connectivity](https://github.com/AmingWu/CCN/blob/master/pic/fig3.png?raw=true "Illustration of Visual Neuron Connectivity")

## Setting Up and Data Preparation
We used pytorch 1.1.0, python 3.6, and CUDA 9.0 for this project. Before using this code, you should download VCR dataset from this link, i.e., https://visualcommonsense.com/. Follow the steps given by the link, i.e., https://github.com/rowanz/r2c/, to set up the running environment.

## Training and Validation
export CUDA_VISIBLE_DEVICES=0,1,2    
python train.py -params multiatt/default.json -folder saves/flagship_answer

## Citation
```bibtex
@incollection{NIPS2019_8804,
title = {Connective Cognition Network for Directional Visual Commonsense Reasoning},
author = {Wu, Aming and Zhu, Linchao and Han, Yahong and Yang, Yi},
booktitle = {Advances in Neural Information Processing Systems 32},
url = {http://papers.nips.cc/paper/8804-connective-cognition-network-for-directional-visual-commonsense-reasoning.pdf}
}
```
