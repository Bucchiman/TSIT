#!/usr/bin/env bash

set -x

NAME='ast_green2cherry'
TASK='AST'
DATA='green2cherry'
CROOT='./datasets/green2cherry'
SROOT='./datasets/green2cherry'
CKPTROOT='./checkpoints'
WORKER=4

python train.py \
    --name $NAME \
    --task $TASK \
    --gpu_ids 0 \
    --checkpoints_dir $CKPTROOT \
    --batchSize 1 \
    --dataset_mode $DATA \
    --croot $CROOT \
    --sroot $SROOT \
    --nThreads $WORKER \
    --no_pairing_check \
    --no_instance \
    --gan_mode hinge \
    --num_upsampling_layers more \
    --alpha 1.0 \
    --display_freq 50 \
    --save_epoch_freq 20 \
    --niter 100 \
    --lambda_vgg 1 \
    --lambda_feat 1 \
    --crop_size 1024 \
    --aspect_ratio 2.0 \
    --save_epoch_freq 1000
