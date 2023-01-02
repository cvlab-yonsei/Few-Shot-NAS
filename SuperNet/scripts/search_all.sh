echo script name: $0

dataset=$1
NUM_GPU=$2
SPLIT=$3

channel=16
num_cells=5
max_nodes=4
space=nas-bench-201

data_path="../data/cifar.python"

save_dir=./SuperNet
OUTPUT=./SuperNet/logs
#CKPT=seed-0-opt5-wotrash/seed-0-last.pth
CKPT=seed-5349-opt5-wotrash-baseline-250/seed-5349-last.pth
#CKPT=seed-5983-opt5-wotrash-decom5-5-15-20-128-750/seed-5983-last.pth
BN=0

CUDA_VISIBLE_DEVICES=${NUM_GPU} OMP_NUM_THREADS=4 python ./SuperNet/search_all.py \
    --data_path ${data_path} --dataset ${dataset} \
    --search_space_name ${space} --max_nodes ${max_nodes} \
    --channel ${channel} --num_cells ${num_cells} \
    --track_running_stats ${BN} \
    --config_path SuperNet/configs/OneShotFromAutoDL.config \
    --workers 4 --save_dir ${save_dir} \
    --rand_seed 0 \
    --output_dir ${OUTPUT} --edge_op ${SPLIT} \
    --ckpt ${CKPT}
