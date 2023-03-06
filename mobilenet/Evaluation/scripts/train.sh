echo script name: $0

TAG=$1
SEED=$2

NUM_GPU=8

python ./Evaluation/train.py --tag ${TAG} --seed ${SEED} --num_gpus ${NUM_GPU} \
    --workers 3 --nesterov \
    --interval_ep_eval 240 \
    --arch 2 1 0 6 4 2 4 4 4 0 0 4 3 2 6 0 4 2 2 4 3
