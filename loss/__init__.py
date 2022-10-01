from torch.nn import CrossEntropyLoss
from loss.dmml import DMMLLoss

def make_loss(args, gids):
    gid = None if gids is None else gids[0]
    if args.loss_type == 'dmml':
        criterion = DMMLLoss(args=args, num_support=args.num_support, distance_mode=args.distance_mode,
                             margin=args.margin, gid=gid)
    else:
        raise NotImplementedError

    return criterion
