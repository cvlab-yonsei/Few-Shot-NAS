##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
######################################################################################
# One-Shot Neural Architecture Search via Self-Evaluated Template Network, ICCV 2019 #
######################################################################################
import os, sys, time, random, argparse, json
from copy import deepcopy
import torch
import numpy as np
from pathlib import Path

from nas_201_api  import NASBench201API as API
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from datasets.get_dataset_with_transform import get_datasets, get_nas_search_loaders
from models.OneShot import UniformRandomSupernet
from models.cell_operations import SearchSpaceNames
from utils.genotypes import Structure
from utils.flop_benchmark import get_model_infos
from utils.meter import AverageMeter
from utils.evaluation_utils import obtain_accuracy
from utils.config_utils import load_config
from utils.starts import prepare_seed, prepare_logger


import tqdm
max_train_iters = 100

#def search_find_best(train_loader, valid_loader, network, n_samples, logger):
#    with torch.no_grad():
#        train_iter = iter(train_loader)
#        valid_iter = iter(valid_loader)
#        archs = []
#        valid_accs = []
#        for i in range(n_samples):
#            logger.log("[{:5d}/{:5d}] Clear BN statistics".format(i, n_samples))
#            for m in network.modules():
#                if isinstance(m, torch.nn.BatchNorm2d):
#                    m.track_running_stats = True
#                    m.running_mean = torch.nn.Parameter(torch.zeros(m.num_features, device="cuda"), requires_grad=False)
#                    m.running_var = torch.nn.Parameter(torch.ones(m.num_features, device="cuda"), requires_grad=False)
#        
#            arch = network.random_genotype()
#            archs.append(arch)
#
#            logger.log("Calibrating BNs of {}".format(arch))
#            network.train()
#            for step in tqdm.tqdm(range(max_train_iters)):
#                try:
#                    base_inputs,_,_,_ = next(train_iter)
#                except:
#                    train_iter = iter(train_loader)
#                    base_inputs,_,_,_ = next(train_iter)
#
#                output, logits = network(base_inputs.cuda(non_blocking=True), arch)
#                del base_inputs, output, logits
#
#
#            network.eval()
#            try:
#                inputs, targets = next(valid_iter)
#            except:
#                valid_iter = iter(valid_loader)
#                inputs, targets = next(valid_iter)
#
#            _, logits = network(inputs.cuda(non_blocking=True), arch) # ADDED cuda
#            val_top1, val_top5 = obtain_accuracy(logits.cpu().data, targets.data, topk=(1, 5))
#
#            logger.log("Top1={:.2f}%\n".format(val_top1.item()))
#            valid_accs.append(val_top1.item())
#
#        best_idx = np.argmax(valid_accs)
#        best_arch, best_valid_acc = archs[best_idx], valid_accs[best_idx]
#        return best_arch, best_valid_acc


def search_find_best(xloader, network, n_samples, logger):
    with torch.no_grad():
        network.eval()
        archs, valid_accs = [], []
        loader_iter = iter(xloader)
        for i in range(n_samples):
            arch = network.random_genotype()
            try:
                inputs, targets = next(loader_iter)
            except:
                loader_iter = iter(xloader)
                inputs, targets = next(loader_iter)

            _, logits = network(inputs.cuda(non_blocking=True), arch) # ADDED cuda
            val_top1, val_top5 = obtain_accuracy(logits.cpu().data, targets.data, topk=(1, 5))

            archs.append(arch)
            valid_accs.append(val_top1.item())
            logger.log("[{:5d}/{:5d}] Top1={:.2f}% from {}".format(i, n_samples, val_top1.item(), arch))

        best_idx = np.argmax(valid_accs)
        best_arch, best_valid_acc = archs[best_idx], valid_accs[best_idx]
        return best_arch, best_valid_acc



def search_find_best_all(xloader, network, n_samples, logger):
    with torch.no_grad():
        network.eval()
        archs, valid_accs = [], []
        for i in range(n_samples):
            #arch = network.module.random_genotype(True)
            arch = network.random_genotype()
            arch_top1, arch_top5 = AverageMeter(), AverageMeter()
            for inputs, targets in xloader:
                _, logits = network(inputs.cuda(non_blocking=True), arch) # ADDED cuda
                val_top1, val_top5 = obtain_accuracy(logits.cpu().data, targets.data, topk=(1, 5))
                arch_top1.update(val_top1.item(), inputs.size(0))
                arch_top5.update(val_top5.item(), inputs.size(0))

            archs.append(arch)
            valid_accs.append(arch_top1.avg)
            logger.log("[{:5d}/{:5d}] Top1={:.2f}% from {}".format(i, n_samples, arch_top1.avg, arch))

        best_idx = np.argmax(valid_accs)
        best_arch, best_valid_acc = archs[best_idx], valid_accs[best_idx]
        return best_arch, best_valid_acc


def main(xargs):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads(xargs.workers)
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
  config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
  test_batch_size = 512
  #valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=test_batch_size, shuffle=False, pin_memory=True, num_workers=0)
  train_loader, _, valid_loader = get_nas_search_loaders(
      train_data,
      valid_data,
      xargs.dataset,
      "SuperNet/configs/",
      #(test_batch_size, test_batch_size),
      (config.batch_size, test_batch_size),
      0 #xargs.workers,
  )
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))
  logger.log('||||||| {:10s} ||||||| Train-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(train_loader), len(valid_loader), test_batch_size))

  search_space = SearchSpaceNames[xargs.search_space_name]
  search_model = UniformRandomSupernet(
      C=xargs.channel, N=xargs.num_cells, max_nodes=xargs.max_nodes,
      num_classes=class_num, search_space=search_space, 
      affine=False,
      track_running_stats=bool(xargs.track_running_stats)
  )
  
  flop, param  = get_model_infos(search_model, xshape)
  logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
  if xargs.arch_nas_dataset is None:
      api = None
  else:
      api = API(xargs.arch_nas_dataset)
  logger.log('create API = {:} done'.format(api))

  #search_model = torch.nn.DataParallel(search_model).cuda()
  search_model = search_model.cuda()

  ckpt_path = logger.model_dir / xargs.ckpt
  logger.log("=> loading checkpoint of '{:}'".format(ckpt_path))
  checkpoint = torch.load(ckpt_path)['search_model']
  search_model.load_state_dict(checkpoint)

  start_time = time.time()
  best_arch, best_acc = search_find_best(valid_loader, search_model, xargs.select_num, logger) 
  #best_arch, best_acc = search_find_best(train_loader, valid_loader, search_model, xargs.select_num, logger) 
  #best_arch, best_acc = search_find_best_all(valid_loader, search_model, xargs.select_num, logger) 
  search_time = time.time() - start_time
  logger.log("The best one : {:} with accuracy={:.2f}%, with {:.1f} s.".format(best_arch, best_acc, search_time))

  if api is not None:
      logger.log("{:}".format(api.query_by_arch(best_arch, "200")))
  logger.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser("SETN")
  parser.add_argument('--data_path',          type=str,   help='Path to dataset')
  parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  # channels and number-of-cells
  parser.add_argument('--search_space_name',  type=str,   help='The search space name.')
  parser.add_argument('--max_nodes',          type=int,   help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   help='The number of cells in one stage.')
  parser.add_argument('--select_num',         type=int,   help='The number of selected architectures to evaluate.')
  parser.add_argument('--track_running_stats',type=int,   choices=[0,1],help='Whether use track_running_stats or not in the BN layer.')
  parser.add_argument('--config_path',        type=str,   help='The path of the configuration.')
  # log
  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--arch_nas_dataset',   type=str,   help='The path to load the architecture dataset (tiny-nas-benchmark).')
  parser.add_argument('--rand_seed',          type=int,   help='manual seed')
  parser.add_argument('--edge_op',            type=int,   help='index of the operation of the edge')
  parser.add_argument('--ckpt',               type=str,   help='pre-trained SuperNet')

  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  main(args)
