from torchvision import transforms
from torch.utils.data import dataloader
import numpy as np
import random
from data.CL_market import Market1501
from data.sampler import RandomSampler
from data.random_erasing import RandomErasing
import numpy
import torch
import pdb

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def make_dataloader(args, task_id, epoch=0):
    ######################
    gen = torch.Generator()
    gen.manual_seed(args.manual_seed)
    #################
    train_list = [
        transforms.Resize((args.img_height, args.img_width), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    if args.random_erasing:
        probability = 0.3 + 0.4 * min((float(epoch) / args.num_epochs), 0.8)
        s_epoch = 0.1 + 0.3 * min((float(epoch) / args.num_epochs), 0.8)
        train_list.append(RandomErasing(probability=probability, s_epoch=s_epoch, mean=[0.0, 0.0, 0.0]))
    train_transform = transforms.Compose(train_list)

    batch_m = args.num_classes
    if 'dmml' in args.loss_type:
        batch_k = args.num_support + args.num_query
    else:
        batch_k = args.num_instances

    if args.dataset == 'market1501':
        train_set = Market1501(args.dataset_root, train_transform, task_id=task_id,
                               split='train', ROOT_PATH=args.preprocess_data_path, 
                               dataset_name='market1501')
    else:
        raise NotImplementedError

    ######## if torch version >= 1.6.0, we need generator to fix seed 
    ####### also GPU mode also makes difference
    if torch.__version__ >= '1.6.0':
        # print(torch.__version__)
        train_loader = dataloader.DataLoader(train_set, sampler=RandomSampler(train_set, batch_k, CL_sign=True), \
                                            batch_size=batch_m * batch_k, \
                                            num_workers=args.num_workers, drop_last=True,
                                            worker_init_fn=seed_worker,
                                            generator=gen
                                            )
    else:
        train_loader = dataloader.DataLoader(train_set, sampler=RandomSampler(train_set, batch_k, CL_sign=True), \
                                            batch_size=batch_m * batch_k, \
                                            num_workers=args.num_workers, drop_last=True,
                                            worker_init_fn=seed_worker,
                                            # generator=gen
                                            )
    return train_loader
