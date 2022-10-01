#! /usr/bin/env bash
NUM_CLASSES_BATCH=32
START_ID=0
DATASET_NAME='market1501'
IMG_HEIGHT=256
IMG_WIDTH=128
METRIC='euclidean'
EPOCH=600
HALF_EPOCH=300
TEMPERATURE=10.0

for SEED in {0..10};
do

echo "seed is set to $SEED for reproducing"

python CL_train_DwoPP.py \
--dataset ${DATASET_NAME} \
--dataset_root datasets/${DATASET_NAME} \
--exp_root dmml/${DATASET_NAME} \
--lr=2e-4 \
--num_epochs $EPOCH \
--lr_decay_start_epoch $HALF_EPOCH \
--weight_decay=1e-4 \
--num_classes $NUM_CLASSES_BATCH \
--distance_mode='hard_mining' \
--num_support=5 \
--num_query=1 \
--margin=0.4 \
--img_height $IMG_HEIGHT \
--img_width $IMG_WIDTH \
--num_workers=24 \
--gpu='2,3' \
--random_erasing \
--remove_downsample \
--cuda \
--method DwPP_seed_${SEED} \
--loss_type dmml \
--preprocess_data_path preprocess_dataset/ \
--start_task_id $START_ID \
--weight_knowledge_distill 1.0 \
--dmml_dist_metric ${METRIC} \
--distillation_dist_metric ${METRIC} \
--manual_seed $SEED \
--temperature $TEMPERATURE \
# --remove_positive_pair \
done