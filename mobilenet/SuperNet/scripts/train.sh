echo script name: $0

TAG=$1
SEED=$2

NUM_GPU=8

python ./SuperNet/train.py --tag ${TAG} --seed ${SEED} --num_gpus ${NUM_GPU} \
    --workers 3 --nesterov
