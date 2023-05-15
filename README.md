# Portable Quantum Language Model

## Introduction
This is the repository for the paper: [PQLM - Multilingual Decentralized Portable Quantum Language Model](https://ieeexplore.ieee.org/document/10095215).

We introduce a portable quantum language model consisted of random quantum circuits and trained on NISQ machines, in which we propose a small quantum LSTM language model that is highly portable (meaning, information contained in it can be easily transferred to classical machines) for downstream applications.

Our PQLM exhibits comparable performance to its classical counterpart on both intrinsic evaluation (loss, perplexity) and extrinsic evaluation (multilingual sentiment analysis accuracy) metrics.


## Citation
To cite PQLM in your work, please include the following bibtex reference:
```
@INPROCEEDINGS{10095215,
  author={Li, Shuyue Stella and Zhang, Xiangyu and Zhou, Shu and Shu, Hongchao and Liang, Ruixing and Liu, Hexin and Garcia, Leibny Paola},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={PQLM - Multilingual Decentralized Portable Quantum Language Model}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10095215}
}
```