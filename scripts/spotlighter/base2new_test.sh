#!/bin/bash

# custom config
DATA=DATA
TRAINER=Spotlighter

DATASET=$1
SEED=$2

CFG=vit_b16_ep10_ctxv1
SHOTS=16
LOADEP=20        
SUB=new

CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
CSC=False  # class-specific context (False or True)

COMMON_DIR=${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_cscFalse_ctpend/seed${SEED}/
MODEL_DIR=output/base2new/${COMMON_DIR}
DIR=output/base2new/test_${SUB}/${COMMON_DIR}

if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    CUDA_VISIBLE_DEVICES=3 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi