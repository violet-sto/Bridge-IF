export CUDA_VISIBLE_DEVICES=0
# or use multi-gpu training when you want:
# export CUDA_VISIBLE_DEVICES=0,1

model=bridge_if_esm1b_650m_pifold
exp=fixedbb/${model}
dataset=cath_4.2
name=fixedbb/${dataset}/${model}

python ./train.py \
    experiment=${exp} datamodule=${dataset} name=${name} \
    task.generator.diffusion_steps=25 \
    logger=wandb trainer=ddp_fp16