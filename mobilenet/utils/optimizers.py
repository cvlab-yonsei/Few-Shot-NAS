import torch.optim as optim


def get_optimizer_scheduler(args, model):
    optimizer = optim.SGD(
        get_params(model),	
        lr=args.learning_rate, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay,
        nesterov=args.nesterov, 
    )

    if args.lr_schedule_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iter, eta_min=0)
    elif args.lr_schedule_type == 'linear':
        lr_lambda = lambda it: (1-it/args.max_iter) 
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda) 
    elif args.lr_schedule_type == 'poly':
        lr_lambda = lambda it: (1-it/args.max_iter)**0.9
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda) 
    else:
        raise NotImplementedError
    return optimizer, scheduler


def get_params(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            #print('w/  WD : ', pname, p.size())
            group_weight_decay.append(p)
        else:
            #print('w/o WD : ', pname, p.size())
            group_no_weight_decay.append(p)

    assert len(list(model.parameters())) == len(group_weight_decay) + len(group_no_weight_decay)
    return [dict(params=group_weight_decay), dict(params=group_no_weight_decay, weight_decay=0.)]
