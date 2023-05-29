# ABGNN

The work of ``[Pre-training Antibody Language Models for Antigen-Specific Computational Antibody Design]()''.

## Introduction

<p align="center"><img src="figures/framework.png" width=80%></p>
<p align="center"><b>Schematic illustration of the ABGNN framework</b></p>

The AbBERT is the pre-trained antibody model. Its `soft' prediction will be fed into the sequence GNN $\mathcal{H}_{seq}$, after encoding and generating the updated sequence, structure GNN $\mathcal{H}_{str}$ encodes the updated graph and then predict the structures. The sequence and structure prediction iteratively refine $T$ times. We use AA as the abbreviation of amino acid.

## Dependencies

- pytorch==1.12.0
- fairseq==0.10.2
- numpy==1.23.3

## Pretrain a model on sequence data

We collected all paired and unpaired data from [OAS Database](https://opig.stats.ox.ac.uk/webapps/oas/oas) using the provided scripts. We extracted the antibody sequences along with their CDR tags. Then we randomly split the dataset into three subsets: 1000 for validation, 1000 for testing, and the remaining for training. After processing, we obtained the following files: `seq.train.tokens`, `seq.valid.tokens`, `seq.test.tokens` and corresponding `tag.train.tokens`, `tag.valid.tokens`, `tag.test.tokens`. Finally, we preprocess these files into fairseq binary files using following scripts.

```shell
bash pretrain-preprocess.sh
```

When training, we can run:

```shell
bash pretrain-abbert.sh
```

## Finetune on sequence and structure co-design tasks

For experiment 1, we refer to the preprocessing scripts in [MEAN](https://github.com/THUNLP-MT/MEAN) and convert it to jsonl files, similar to experiment 2. For experiment 2, We directly use data from [HSRN](https://github.com/wengong-jin/abdockgen). For experiment 3, we follow the setting in [RefineGNN](https://github.com/wengong-jin/RefineGNN).

The finetuning scripts are following:

```shell
bash finetune-exp1.sh

bash finetune-exp2.sh

bash covid-optimize.sh
```

## License

This work is under [MIT License](LICENSE)

## Citation

If you find this code useful in your research, please consider citing:
