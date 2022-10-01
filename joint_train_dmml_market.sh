#! /usr/bin/env bash
NUM_CLASSES_BATCH=32
DATASET_NAME='market1501' 
LOSS='dmml'
IMG_HEIGHT=256  
IMG_WIDTH=128 
METRIC='euclidean'
EPOCH=600
HALF_EPOCH=300

for SEED in {0..10};
do
echo "seed is set to $SEED for reproducing"

python train.py \
--dataset ${DATASET_NAME} \
--dataset_root ./datasets/${DATASET_NAME} \
--exp_root ./dmml/${DATASET_NAME} \
--lr=2e-4 \
--num_epochs $EPOCH \
--lr_decay_start_epoch $HALF_EPOCH \
--weight_decay=1e-4 \
--num_classes $NUM_CLASSES_BATCH \
--distance_mode='hard_mining' \
--num_support=5 \
--num_query=1 \
--num_instances 6 \
--margin=0.4 \
--img_height $IMG_HEIGHT \
--img_width $IMG_WIDTH \
--num_workers=24 \
--gpu='6,7' \
--cuda \
--loss_type ${LOSS} \
--method  ${LOSS}_${METRIC}_seed_${SEED} \
--remove_downsample \
--random_erasing \
--dmml_dist_metric ${METRIC} \
--manual_seed $SEED

done