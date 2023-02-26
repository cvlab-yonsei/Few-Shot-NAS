import os
import sys
import time
import logging
import argparse
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
### from Detectron2 ###
import utils.comm as comm
from utils.engine import launch
from utils.distributed_sampler import seed_all_rng 

from utils.datasets import get_datasets
from utils.optimizers import get_optimizer_scheduler
from utils.losses import get_losses
from utils.evaluator import Evaluator
from models.OneShot import SuperNet
from models.layers import SearchSpaceNames


def arch_uniform_sampling(choices, distributed=True):
    rand_arch = []
    for i, ops in enumerate(choices):
        rand_arch += [ np.random.choice(ops) ]
#    if distributed:
#        rand_arch = torch.tensor(rand_arch).to(torch.device("cuda"))
#        torch.distributed.broadcast(rand_arch, 0)
    return rand_arch


def do_train(args, model, logger):
    #trainset, validset, train_loader, valid_loader = get_datasets(args)
    trainset, validset = get_datasets(args)
    logger.info("Trainset Size: {:7d}".format(len(trainset)))
    logger.info("Validset Size: {:7d}".format(len(validset)))

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    iters_per_epoch = len(trainset) // args.train_batch_size
    args.max_iter   = iters_per_epoch * args.max_epoch

    optimizer, scheduler = get_optimizer_scheduler(args, model)
    criterion = get_losses(args).to(torch.device("cuda"))
    CHOICES = deepcopy(model.module.choices)

    logger.info(f"--> START {args.save_name}")
    model.train()
    storages = {"CE": 0}
    interval_iter_verbose = iters_per_epoch // 10
    for ep in range(args.max_epoch):
        for it, (img, gt) in enumerate(train_loader):
            rand_arch = arch_uniform_sampling(CHOICES)

            logits = model(img.to(torch.device("cuda")), rand_arch)
            loss   = criterion(logits, gt.to(torch.device("cuda")))
            optimizer.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step() 
            storages["CE"] += loss.item()

            if it % interval_iter_verbose == 0:
                verbose = f"iter: {it:5d}/{iters_per_epoch:5d}  CE: {loss.item():.4f}  "
                logger.info(verbose)

        for k in storages.keys(): storages[k] /= iters_per_epoch
        verbose = f"--> epoch: {ep:3d}/{args.max_epoch:3d}  avg CE: {storages['CE']:.4f}  lr: {scheduler.get_last_lr()[0]}  "
        logger.info(verbose)
        for k in storages.keys(): storages[k] = 0

    if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
        ckpt = model.module.state_dict()
    else:
        ckpt = model.state_dict()
    info = {"state_dict": ckpt, "args": args}
    torch.save(info, args.ckpt_path)
    logger.info(f"--> END {args.save_name}")


def main(args):
    args.save_name = f"{args.tag}-seed-{args.seed}"
    args.log_path  = f"{args.save_path}/logs/{args.save_name}.txt"
    args.ckpt_path = f"{args.save_path}/checkpoint/{args.save_name}.pt"

    logger = logging.getLogger("SuperNet Training")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler(args.log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    for arg in vars(args):
        logger.info(f'{arg:<20}: {getattr(args, arg)}')

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if args.seed < 0 else args.seed + comm.get_rank())

    search_space = SearchSpaceNames[args.search_space]
    model = SuperNet(search_space, affine=True, track_running_stats=True, freeze_bn=args.freeze_bn)
    if args.num_gpus > 1:
        if not args.freeze_bn: torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.DataParallel(model)
    model.to(torch.device("cuda"))

    start_time = time.time()
    do_train(args, model, logger)
    end_time = time.time() - start_time
    hours = int(end_time // 3600)
    mins  = int((end_time % 3600) // 60)
    logger.info(f"ELAPSED TIME: {end_time:.1f}(s) = {hours:02d}(h) {mins:02d}(m)")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tag', type=str)
    parser.add_argument('--seed', type=int, default=-1) 
    parser.add_argument('--workers', type=int, default=4) 

    parser.add_argument('--data_path', type=str, default='../../data/imagenet')
    parser.add_argument('--save_path', type=str, default='./SuperNet')
    parser.add_argument('--search_space', type=str, default='proxyless', choices=['proxyless', 'spos', 'greedynas-v1'])
    parser.add_argument('--valid_size', type=int, default=50000, choices=[None, 50000])

    parser.add_argument("--num_gpus", type=int, default=2, help="the number of gpus")
    parser.add_argument('--interval_ep_eval', type=int, default=8)

    parser.add_argument('--train_batch_size', type=int, default=512) 
    parser.add_argument('--test_batch_size', type=int, default=256) 
    parser.add_argument('--max_epoch', type=int, default=120) 
    parser.add_argument('--learning_rate', type=float, default=0.12) 
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=4e-5)
    parser.add_argument('--nesterov', default=False, action='store_true')
    parser.add_argument('--lr_schedule_type', type=str, default='cosine', choices=['linear', 'poly', 'cosine'])

    parser.add_argument('--freeze_bn', default=False, action='store_true')
    parser.add_argument('--label_smooth', type=float, default=0.1)
    #parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
