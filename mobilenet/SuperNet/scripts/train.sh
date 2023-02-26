echo script name: $0

TAG=$1
SEED=$2

NUM_GPU=2
SEARCH_SPACE='proxyless'

CUDA_VISIBLE_DEVICES=0,1 python ./SuperNet/train_dp.py --tag ${TAG} --seed ${SEED} --num_gpus ${NUM_GPU} \
    --nesterov --freeze_bn
