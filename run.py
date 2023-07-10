import argparse
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn
# import torch.distributed as dist
# import torch.nn.parallel as parallel
import torch.utils.data as data

import dataset as ds
import engine
from models import Model


def get_args_parser():
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', default=19970423, type=int)
    parser.add_argument('--root', default='../../DataSet/ASAP', type=str)
    parser.add_argument('--prompt_idx', default=1, type=int, help='{1, 2, 3, 4, 5, 6, 7, 8}')

    parser.add_argument('--weight_init', default=0.9, type=float)

    parser.add_argument('--training_type', default='inductive', type=str, help='{inductive, transductive}')

    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.00005, type=float)
    parser.add_argument('--memory_lr', default=0.05, type=float)
    parser.add_argument('--weight_decay', default=0.0005, type=float)

    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(args):
    num_classes = ds.NUM_SCORES[args.prompt_idx]
    train_dataset, dev_dataset, test_dataset = ds.load_datasets(args.root, args.prompt_idx)

    num_features = 20
    model = Model(num_features, args.weight_init).cuda()
    # model = parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    optim_parameters = [
        {'params': [p for n, p in model.named_parameters() if not n.endswith('weight_memory') and p.requires_grad], 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if n.endswith('weight_memory') and p.requires_grad], 'lr': args.memory_lr, 'weight_decay': args.weight_decay}
    ]
    optimizer = torch.optim.AdamW(optim_parameters)

    if args.training_type == 'transductive':
        dataset = data.ConcatDataset([train_dataset, dev_dataset, test_dataset])
        # train_dist_sampler = data.distributed.DistributedSampler(dataset)
        train_dataloader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
        test_dataloader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False)
    elif args.training_type == 'inductive':
        # train_dist_sampler = data.distributed.DistributedSampler(train_dataset)
        train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=True, shuffle=True)
        test_dataloader = data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False)
    else:
        raise NotImplementedError

    min_ep_loss = 100
    best_reg_qwk, best_u_qwk, best_gt_qwk, best_t_qwk, best_n_qwk = 0, 0, 0, 0, 0
    for ep_idx in range(args.num_epochs):
        # train_dist_sampler.set_epoch(ep_idx)

        ep_loss, train_t = engine.train(model, optimizer, train_dataloader)

        # dist.reduce(ep_loss, dst=args.master_rank, op=dist.ReduceOp.SUM)
        # if args.local_rank == args.master_rank:
        # ep_loss = ep_loss / dist.get_world_size()
        gt_qwk, u_qwk, reg_qwk, t_qwk, n_qwk, test_t = engine.evaluate(model, test_dataloader, num_classes)
        info = f'P{args.prompt_idx};Epoch={ep_idx + 1}/{args.num_epochs};TrainT={train_t:.1f}s;TestT={test_t:.1f}s;L={ep_loss:.4f};QWK={reg_qwk:.4f};'
        if ep_loss <= min_ep_loss:
            min_ep_loss = ep_loss
            best_reg_qwk, best_u_qwk, best_gt_qwk, best_t_qwk, best_n_qwk = reg_qwk, u_qwk, gt_qwk, t_qwk, n_qwk
            torch.save(model.state_dict(), f'{args.training_type}_P{args.prompt_idx}.pkl')
        info += f'minL={min_ep_loss:.4f};BestQWK={best_reg_qwk:.4f}.'
        print(info)

    info = f'     R,      G,      U,      T,      N | FINAL | prompt {args.prompt_idx}\n'
    info += f'{best_reg_qwk:.4f}, {best_gt_qwk:.4f}, {best_u_qwk:.4f}, {best_t_qwk:.4f}, {best_n_qwk:.4f}\n'
    print(info)


def __main__():
    args = get_args_parser()
    # dist.init_process_group(backend='nccl')
    # set_random_seed(args.random_seed + args.local_rank)
    set_random_seed(args.random_seed)
    # torch.cuda.set_device(torch.device(f'cuda:{args.local_rank}'))
    run(args)


if __name__ == '__main__':
    __main__()
