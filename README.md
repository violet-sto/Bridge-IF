# Bridge-IF
Official implementation of NeurIPS'24 paper "[Bridge-IF: Learning Inverse Protein Folding with Markov Bridges](https://arxiv.org/abs/2411.02120)".  This code is built on top of the [LM-Design repo](https://github.com/BytedProtein/ByProt). 

<!-- </div> -->

## Installation

```bash
# clone project
git clone --recursive https://github.com/violet-sto/Bridge-IF.git
cd Bridge-IF

# create conda virtual environment
conda env create -f env.yml
conda activate BridgeIF
pip install -r requirements.txt
```

## Structure-based protein sequence design (inverse folding)
**Pretrained structure encoder** ([Zenodo](https://zenodo.org/records/14809937))

### Data

**Download the preproceesd CATH datasets**
- CATH 4.2 dataset provided by [Generative Models for Graph-Based Protein Design (Ingraham et al, NeurIPS'19)](https://papers.nips.cc/paper/2019/hash/f3a4ff4839c56a5f460c88cce3666a2b-Abstract.html)
- CATH 4.3 dataset provided by [Learning inverse folding from millions of predicted structures (Hsu et al, ICML'22)](https://www.biorxiv.org/content/10.1101/2022.04.10.487779v1) 
```bash
bash scripts/download_cath.sh
```
Go check `configs/datamodule/cath_4.*.yaml` and set `data_dir` to the path of the downloaded CATH data. 

**Dowload PDB complex data (multichain)**

This dataset curated protein (multichain) complexies from Protein Data Bank (PDB). 
It is provided by [Robust deep learning-based protein sequence design using ProteinMPNN](https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1). 
See their [github page](https://github.com/dauparas/ProteinMPNN/blob/main/training/README.md) for more details.
```bash
bash scripts/download_multichain.sh
```
Go check `configs/datamodule/multichain.yaml` and set `data_dir` to the path of the downloaded multichain data. 

<!-- <br> -->


### Training Bridge-IF
In the following sections, we will use CATH 4.2 dataset as an runing example.

```bash
model=bridge_if_esm1b_650m_pifold
exp=fixedbb/${model}
dataset=cath_4.2
name=fixedbb/${dataset}/${model}

python ./train.py \
    experiment=${exp} datamodule=${dataset} name=${name} \
    task.generator.diffusion_steps=25 \
    logger=wandb trainer=ddp_fp16
```

### Evaluation/inference on valid/test datasets

```bash
dataset=cath_4.2
name=fixedbb/${dataset}/bridge_if_esm1b_650m_pifold
exp_path=logs/${name}

python ./test.py \                                                                 
    experiment_path=${exp_path} \
    data_split=test ckpt_path=best.ckpt mode=predict
```                                     

## Citation
```
@inproceedings{
    zhu2024bridgeif,
    title={Bridge-{IF}: Learning Inverse Protein Folding with Markov Bridges},
    author={Yiheng Zhu and Jialu Wu and Qiuyi Li and Jiahuan Yan and Mingze Yin and Wei Wu and Mingyang Li and Jieping Ye and Zheng Wang and Jian Wu},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=Q8yfhrBBD8}
}
```