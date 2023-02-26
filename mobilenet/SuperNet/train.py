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


#def do_test(args, model, logger, testset, test_loader):
#    evaluator = Evaluator(distributed=args.num_gpus > 1) 
#    evaluator.reset()
#    with torch.no_grad():
#        model.eval()
#        for batch in test_loader:
#            img = torch.stack([x[0] for x in batch], dim=0).to(torch.device("cuda"))  
#            gt  = torch.stack([x[1] for x in batch], dim=0).numpy()
#            logits = model(img)
#            pred = logits.argmax(dim=1).to(torch.device("cpu")).numpy() 
#            evaluator.process(pred, gt)
#    results = evaluator.evaluate()
#    logger.info("# of Test Samples: {}".format(results["Total samples"]))
#    logger.info(f"Top-1 acc: {results['Top1_acc']:.2f}  Top-5 acc: {results['Top5_acc']:.2f}")
#    logger.info(f"Top-1 err: {results['Top1_err']:.2f}  Top-5 err: {results['Top5_err']:.2f}")
#    if results is None: results = {}
#    return results


def arch_uniform_sampling(choices, distributed=True):
    rand_arch = []
    for i, ops in enumerate(choices):
        rand_arch += [ np.random.choice(ops) ]
    if distributed:
        rand_arch = torch.tensor(rand_arch).to(torch.device("cuda"))
        torch.distributed.broadcast(rand_arch, 0)
    return rand_arch


def do_train(args, model, logger):
    trainset, validset, train_loader, valid_loader = get_datasets(args)
    logger.info("Trainset Size: {:7d}".format(len(trainset)))
    logger.info("Validset Size: {:7d}".format(len(validset)))

    iters_per_epoch = len(trainset) // args.train_batch_size
    args.max_iter   = iters_per_epoch * args.max_epoch

    optimizer, scheduler = get_optimizer_scheduler(args, model)
    criterion = get_losses(args).to(torch.device("cuda"))
    CHOICES = deepcopy(model.module.choices)

    logger.info(f"--> START {args.save_name}")
    model.train()
    ep = 1
    storages = {"CE": 0}
    interval_iter_verbose = 1 + iters_per_epoch // 10
    for it, batch in zip(range(1, args.max_iter+1), train_loader):
        img = torch.stack([x[0] for x in batch], dim=0).to(torch.device("cuda"))  
        gt  = torch.tensor([x[1] for x in batch]).to(torch.device("cuda")) 

        rand_arch = arch_uniform_sampling(CHOICES)
        logits = model(img, rand_arch)
        loss_dict = {}
        loss_dict["loss_ce"] = criterion(logits, gt)
        losses = sum(loss_dict.values())
        loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        optimizer.zero_grad()
        losses.backward()
        #nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step() 
        storages["CE"] += losses_reduced 

        if it % interval_iter_verbose == 0:
            verbose = f"{it:5d}/{args.max_iter+1:5d}  CE: {loss_dict_reduced['loss_ce']:.4f}  "
            logger.info(verbose)

        if it % iters_per_epoch == 0:
            for k in storages.keys(): storages[k] /= iters_per_epoch
            verbose = f"epoch: {ep:3d}  avg CE: {storages['CE']:.4f}  lr: {scheduler.get_last_lr()[0]}  "
            logger.info(verbose)
            for k in storages.keys(): storages[k] = 0

#            if ep % args.interval_ep_eval == 0:
#                scores = do_test(args, model, logger, validset, valid_loader)
#                model.train()
#                comm.synchronize()
#                logger.info("\n")

            ep += 1

    if comm.is_main_process():
        if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)):
            ckpt = model.module.state_dict()
        else:
            ckpt = model.state_dict()
        info = {"state_dict": ckpt, "args": args}
        torch.save(info, f"{self.save_path}/{name}_last.pt")
    logger.info(f"--> END {args.save_name}")


def main(args):
    args.save_name = f"{args.tag}-seed-{args.seed}"
    args.log_path  = f"{args.save_path}/logs/{args.save_name}.txt"
    args.ckpt_path = f"{args.save_path}/checkpoint/{args.save_name}.pt"

    logger = logging.getLogger("SuperNet Training")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if comm.is_main_process():
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
    model = SuperNet(search_space, affine=True, track_running_stats=True, freeze_bn=args.freeze_bn).to(torch.device("cuda"))
    if args.num_gpus > 1:
        if not args.freeze_bn: torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        #model = DDP(model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=False) 
        #model = DDP(model, device_ids=[comm.get_local_rank()])
        model = DDP(model, device_ids=[comm.get_local_rank()], find_unused_parameters=True) 
        #model = DDP(model, device_ids=[comm.get_local_rank()], output_device=[comm.get_local_rank()]) 
        #model = DDP(model, device_ids=[comm.get_local_rank()], output_device=comm.get_local_rank(), find_unused_parameters=True) 

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
    launch(main, args.num_gpus, args=(args,)) 
