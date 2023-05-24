# ABGNN

The work of ``[Pre-training Antibody Language Models for Antigen-Specific Computational Antibody Design]()''.

## Introduction

<p align="center"><img src="figures/framework.pdf" width=80%></p>
<p align="center"><b>Schematic illustration of the ABGNN framework</b></p>

The AbBERT is the pre-trained antibody model. Its `soft' prediction will be fed into the sequence GNN $\mathcal{H}_{seq}$, after encoding and generating the updated sequence, structure GNN $\mathcal{H}_{str}$ encodes the updated graph and then predict the structures. The sequence and structure prediction iteratively refine $T$ times. We use AA as the abbreviation of amino acid.

## Environmental Requirements 

```shell
pip install fairseq==0.10.2
```

## Pretrained Models

## Data

## Getting Started

```shell
bash pretrain-abbert.sh

bash finetune-exp1.sh

bash finetune-exp2.sh

bash covid-optimize.sh
```

## License

This work is under [MIT License]()

## Citation

If you find this code useful in your research, please consider citing:
