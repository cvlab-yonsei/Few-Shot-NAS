echo script name: $0

TAG=$1
SEED=$2

NUM_GPU=8

python ./SuperNet/train_ddp_ewgs2.py --tag ${TAG} --seed ${SEED} --num_gpus ${NUM_GPU} \
    --workers 3 --nesterov
#python ./SuperNet/train.py --tag ${TAG} --seed ${SEED} --num_gpus ${NUM_GPU} \
#    --workers 3 --learning_rate 0.045 #--label_smooth 0 


#TAG='val6-10'
#python ./SuperNet/train_ddp_ewgs.py --tag ${TAG} --seed ${SEED} --num_gpus ${NUM_GPU} \
#    --workers 3 --nesterov
#   
#TAG='val6-11'
#python ./SuperNet/train_ddp_ewgs.py --tag ${TAG} --seed ${SEED} --num_gpus ${NUM_GPU} \
#    --workers 3 --nesterov 
#
#TAG='val6-12'
#python ./SuperNet/train_ddp_ewgs.py --tag ${TAG} --seed ${SEED} --num_gpus ${NUM_GPU} \
#    --workers 3 --nesterov 
#
#TAG='val6-13'
#python ./SuperNet/train_ddp_ewgs.py --tag ${TAG} --seed ${SEED} --num_gpus ${NUM_GPU} \
#    --workers 3 --nesterov  
