import torch
import torch.optim as optim
from torch.nn import DataParallel
import numpy as np
import os
import time
import random
from model import resnet_model
from data import make_dataloader
from loss import make_loss
from config import get_parser
from eval import eval


def init_seed(args, gids):
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    if gids is not None:
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def make_model(args, gids=None):
    model = resnet_model(remove_downsample=args.remove_downsample)

    if gids is not None:
        model = model.cuda(gids[0])
        if len(gids) > 1:
            model = DataParallel(model, gids)

    return model    


def adjust_lr_exp(optimizer, base_lr, epoch, num_epochs, decay_start_epoch):

    if epoch < 1:
        raise Exception('Current epoch number should be no less than 1.')
    if epoch < decay_start_epoch:
        return
    for g in optimizer.param_groups:
        g['lr'] = base_lr * (0.005 ** (float(epoch + 1 - decay_start_epoch)
                                        / (num_epochs + 1 - decay_start_epoch)))
    print('=====> lr adjusted to {:.9f}'.format(g['lr']).rstrip('0'))


def train(args, model, optimizer, criterion, gids=None):

    model.train()

    train_loss = []
    t0 = int(time.time())

    for epoch in range(args.num_epochs):
        if epoch % 10 == 0:
            dataloader = make_dataloader(args, epoch)
        print('=== Epoch {}/{} ==='.format(epoch, args.num_epochs))
        adjust_lr_exp(optimizer, args.lr, epoch+1, args.num_epochs, args.lr_decay_start_epoch)

        for iteration, (image, label) in enumerate(dataloader):
            if args.cuda:
                image, label = image.cuda(gids[0]), label.cuda(gids[0])

            feat = model(image)
            loss = criterion(feat, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print training info
            train_loss.append(loss.item())

            if args.loss_type == 'dmml':
                print('Episode: {}, Loss: {:.6f}'.format(iteration, loss.item()))
            else:
                print('Batch: {}, Loss: {:.6f}'.format(iteration, loss.item()))

        avg_training_loss = np.mean(train_loss)
        print('Average loss: {:.6f}'.format(avg_training_loss))
        train_loss = []

        t = int(time.time())
        print('Time elapsed: {}h {}m'.format((t - t0) // 3600, ((t - t0) % 3600) // 60))

    model_save_path = os.path.join(args.exp_root, args.method, 'model_last.pth'.format(epoch))
    if gids is not None and len(gids) > 1:
        torch.save(model.module.state_dict(), model_save_path)
    else:
        torch.save(model.state_dict(), model_save_path)
    print('Final model saved.')

    mAP, CMC=eval(gid=gids[0], dataset=args.dataset, dataset_root=args.dataset_root, 
                which='last', exp_dir= os.path.join(args.exp_root, args.method),\
                    method='joint',verbose=True, bnneck=args.bnneck)
    return mAP, CMC[0].item()


    
def main():
    args = get_parser().parse_args()

    if not os.path.exists(args.exp_root):
        os.makedirs(args.exp_root)

    if torch.cuda.is_available() and not args.cuda:
        print("\nStrongly recommend to run with '--cuda' if you have a device with CUDA support.")

    # print configs
    print('='*40)
    print('Dataset: {}'.format(args.dataset))
    print('Model: ResNet-50')
    print('Optimizer: Adam')
    print('Image height: {}'.format(args.img_height))
    print('Image width: {}'.format(args.img_width))
    print('Loss: {}'.format(args.loss_type))
    if args.loss_type in ['dmml']:
        print('  margin: {}'.format(args.margin))
    print('  class number: {}'.format(args.num_classes))
    if args.loss_type == 'npair':
        pass
    elif args.loss_type == 'dmml':
        print('  support number: {}'.format(args.num_support))
        print('  query number: {}'.format(args.num_query))
        print('  distance_mode: {}'.format(args.distance_mode))
    else:
        print('  instance number: {}'.format(args.num_instances))
    print('Epochs: {}'.format(args.num_epochs))
    print('Learning rate: {}'.format(args.lr))
    print('  decay beginning epoch: {}'.format(args.lr_decay_start_epoch))
    print('Weight decay: {}'.format(args.weight_decay))
    if args.cuda:
        print('GPU(s): {}'.format(args.gpu))
    print('='*40)

    ################### Initialization
    print('Initializing...')
    if args.cuda:
        gpus = ''.join(args.gpu.split())
        gids = [int(gid) for gid in gpus.split(',')]
    else:
        gids = None

    if not os.path.exists(os.path.join(args.exp_root, args.method)):
        os.makedirs(os.path.join(args.exp_root, args.method))

    ############## if seed is not given, randomly generate a seed
    if args.manual_seed == None:
        args.manual_seed=int(time.time())

    args.manual_seed=int(args.manual_seed)
    init_seed(args, gids)
    print(f'seed is set to {args.manual_seed}.')
    #########################################
    model = make_model(args, gids)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = make_loss(args, gids)
    print('Done.')

    ########### writing to file
    text_file = os.path.join('./result', args.dataset, args.method + '.txt')
    f = open(text_file, 'a')
    print(args,file=f)
    f.close()
    ###########Training
    print('Starting training...')
    mAP, Rank_1 = train(args, model, optimizer, criterion, gids)
    print('Training completed.')
    print('the mAP is {:.4f} and Rank-1 is {:.4f}'.format(mAP, Rank_1))

    text_file = os.path.join('./result', args.dataset, args.method + '.txt')
    f = open(text_file, 'a')
    print('{:.4f}, {:.4f}'.format(mAP, Rank_1),file=f)
    print('write to file done!')
    f.close()


if __name__ == '__main__':
    main()
