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


@no_grad_wrapper
def get_cand_err(model, cand, args, train_loader, valid_loader, logger):
  if torch.cuda.is_available():
      device = torch.device('cuda')
  else:
      device = torch.device('cpu')

  max_train_iters = 200 #args.max_train_iters
  max_test_iters = 40 #args.max_test_iters

  logger.log('clear bn statics....')
  for m in model.modules():
    if isinstance(m, torch.nn.BatchNorm2d):
      m.running_mean = torch.zeros_like(m.running_mean)
      m.running_var = torch.ones_like(m.running_var)

  if args.calib_bn:
    logger.log('train bn with training set (BN sanitize) ....')
    model.train()
    for step in tqdm.tqdm(range(max_train_iters)):
      # logger.log('train step: {} total: {}'.format(step,max_train_iters))
      data, target = train_loader.next()
      target = target.type(torch.LongTensor)
      data, target = data.to(device), target.to(device)
      output = model(data, cand)
      del data, target, output
  else:
    logger.log('without calibrating BN ....')

  top1 = 0
  top5 = 0
  total = 0
  logger.log('starting test....')
  model.eval()
  for step in tqdm.tqdm(range(max_test_iters)):
    # logger.log('test step: {} total: {}'.format(step,max_test_iters))
    data, target = valid_loader.next()
    batchsize = data.shape[0]
    target = target.type(torch.LongTensor)
    data, target = data.to(device), target.to(device)
    logits = model(data, cand)
    prec1, prec5 = accuracy(logits, target, topk=(1, 5))
    # logger.log(prec1.item(),prec5.item())
    top1 += prec1.item() * batchsize
    top5 += prec5.item() * batchsize
    total += batchsize
    del data, target, logits, prec1, prec5
  top1, top5 = top1 / total, top5 / total
  top1, top5 = 1 - top1 / 100, 1 - top5 / 100
  logger.log('top1: {:.2f} top5: {:.2f}'.format(top1 * 100, top5 * 100))
  return top1, top5


class EvolutionSearcher(object):
  def __init__(self, args, model, train_loader, valid_loader, logger):
    self.args = args

    self.population_num = args.population_num
    self.select_num = args.select_num
    self.mutation_num = args.mutation_num
    self.crossover_num = args.crossover_num
    self.max_epochs = args.max_epochs
    self.m_prob = args.m_prob
    self.flops_limit = args.flops_limit

    self.model = model
    self.train_loader = train_loader
    self.valid_loader = valid_loader
    self.logger = logger

    self.memory = []
    self.vis_dict = {}
    self.keep_top_k = {self.select_num: [], 50: []}
    self.epoch = 0
    self.candidates = []
    self.nr_layer = 20
    self.nr_state = 4

  def search(self):
    self.logger.log('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))

    self.get_random(self.population_num)
    while self.epoch < self.max_epochs:
      self.logger.log('epoch = {}'.format(self.epoch))
      self.memory.append([])
      for cand in self.candidates:
        self.memory[-1].append(cand)
        self.update_top_k(self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['err'])
        self.update_top_k(self.candidates, k=50, key=lambda x: self.vis_dict[x]['err'])
        self.logger.log('epoch = {} : top {} result'.format(self.epoch, len(self.keep_top_k[50])))
        for i, cand in enumerate(self.keep_top_k[50]):
          self.logger.log('No.{} {} Top-1 err = {}'.format(i + 1, cand, self.vis_dict[cand]['err']))
          ops = [i for i in cand]
          self.logger.log(ops)
          mutation = self.get_mutation(self.select_num, self.mutation_num, self.m_prob)
          crossover = self.get_crossover(self.select_num, self.crossover_num)
          self.candidates = mutation + crossover
          self.get_random(self.population_num)
          self.epoch += 1
    import pdb; pdb.set_trace()
    return self.vis_dict


  def get_random(self, num):
    self.logger.log('random select ........')
    cand_iter = self.stack_random_cand(lambda: tuple(np.random.randint(self.nr_state) for i in range(self.nr_layer)))
    while len(self.candidates) < num:
      cand = next(cand_iter)
      if not self.is_legal(cand):
        continue
      self.candidates.append(cand)
      self.logger.log('random {}/{}'.format(len(self.candidates), num))


  def stack_random_cand(self, random_func, *, batchsize=10):
    while True:
      cands = [random_func() for _ in range(batchsize)]
      for cand in cands:
        if cand not in self.vis_dict:
          self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
      for cand in cands:
        yield cand


  def is_legal(self, cand):
    assert isinstance(cand, tuple) and len(cand) == self.nr_layer
    if cand not in self.vis_dict:
      self.vis_dict[cand] = {}
    info = self.vis_dict[cand]
    if 'visited' in info:
      return False

#    if 'flops' not in info:
#      info['flops'] = get_cand_flops(cand)
#    self.logger.log(cand, info['flops'])
#    if info['flops'] > self.flops_limit:
#      self.logger.log('flops limit exceed')
#      return False

    info['err'] = get_cand_err(self.model, cand, self.args, self.train_loader, self.valid_loader, self.logger)
    info['visited'] = True
    return True


  def update_top_k(self, candidates, *, k, key, reverse=False):
    assert k in self.keep_top_k
    self.logger.log('select ......')
    t = self.keep_top_k[k]
    t += candidates
    t.sort(key=key, reverse=reverse)
    self.keep_top_k[k] = t[:k]


  def get_mutation(self, k, mutation_num, m_prob):
    assert k in self.keep_top_k
    self.logger.log('mutation ......')
    res = []
    iter = 0
    max_iters = 10 * mutation_num

    def random_func():
      cand = list(choice(self.keep_top_k[k]))
      for i in range(self.nr_layer):
        if np.random.random_sample() < m_prob:
          cand[i] = np.random.randint(self.nr_state)
      return tuple(cand)

    cand_iter = self.stack_random_cand(random_func)
    while len(res) < mutation_num and max_iters > 0:
      max_iters -= 1
      cand = next(cand_iter)
      if not self.is_legal(cand):
        continue
      res.append(cand)
      self.logger.log('mutation {}/{}'.format(len(res), mutation_num))
    self.logger.log('mutation_num = {}'.format(len(res)))
    return res


  def get_crossover(self, k, crossover_num):
    assert k in self.keep_top_k
    self.logger.log('crossover ......')
    res = []
    iter = 0
    max_iters = 10 * crossover_num

    def random_func():
      p1 = choice(self.keep_top_k[k])
      p2 = choice(self.keep_top_k[k])
      return tuple(choice([i, j]) for i, j in zip(p1, p2))

    cand_iter = self.stack_random_cand(random_func)
    while len(res) < crossover_num and max_iters > 0:
      max_iters -= 1
      cand = next(cand_iter)
      if not self.is_legal(cand):
        continue
      res.append(cand)
      self.logger.log('crossover {}/{}'.format(len(res), crossover_num))
    self.logger.log('crossover_num = {}'.format(len(res)))
    return res


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
      (config.batch_size, test_batch_size),
      0 #xargs.workers,
  )
  logger.log('||||||| {:10s} ||||||| Valid-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(valid_loader), test_batch_size))
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

  search_space = SearchSpaceNames[xargs.search_space_name]
  logger.log('search space : {:}'.format(search_space))
  search_model = UniformRandomSupernet(
      C=xargs.channel, N=xargs.num_cells, max_nodes=xargs.max_nodes,
      num_classes=class_num, search_space=search_space, affine=False,
      track_running_stats=bool(xargs.track_running_stats)
  )
  
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

  searcher = EvolutionSearcher(args, search_model, train_loader, valid_loader)
  
  start_time = time.time()
  best_arch = searcher.search()
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
  
  parser.add_argument('--calib_bn', type=int, choices=[0,1])
  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  main(args)
