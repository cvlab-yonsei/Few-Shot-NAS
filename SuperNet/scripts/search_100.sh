echo script name: $0

dataset=$1
BN=$2
NUM_GPU=$3
SEED=$4

channel=16
num_cells=5
max_nodes=4
space=nas-bench-201

data_path=../data/cifar.python

benchmark_file=../data/NAS-Bench-201-v1_0-e61699.pth

save_dir=./SuperNet

#CKPT=seed-0-opt5/seed-0-last.pth
CKPT=seed-0-opt5-wotrash/seed-0-last.pth


CUDA_VISIBLE_DEVICES=${NUM_GPU} OMP_NUM_THREADS=4 python ./SuperNet/search_100.py \
    --data_path ${data_path} --dataset ${dataset} \
    --search_space_name ${space} --max_nodes ${max_nodes} \
    --channel ${channel} --num_cells ${num_cells} \
    --select_num 100 \
    --track_running_stats ${BN} \
    --config_path SuperNet/configs/OneShotFromAutoDL.config \
    --workers 4 --save_dir ${save_dir} --arch_nas_dataset ${benchmark_file} \
    --rand_seed ${SEED} \
    --ckpt ${CKPT}
