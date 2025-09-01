#!/bin/bash

# custom config
DATA=/path/to/dataset

TRAINER=Spotlighter

DATASET=imagenet
SEED=$1

CFG=vit_b16_ep10_ctxv1
SHOTS=16


DIR=output/xd/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}/
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    CUDA_VISIBLE_DEVICES=0 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi