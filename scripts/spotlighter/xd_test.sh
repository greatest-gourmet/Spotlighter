# custom config
DATA=DATA

TRAINER=Spotlighter

DATASET=$1
SEED=$2

CFG=vit_b16_ep10_lr8e-4_ctxv1        
SHOTS=16


DIR=output/evaluation/xd/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}/
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
    --model-dir output/xd/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}/ \
    --load-epoch 10 \
    --eval-only
fi
