'''
This file is our method. If all optional losses are set to 0, we have FT+.
'''
import copy
import torch
import torch.optim as optim
from torch.nn import DataParallel
import time
import numpy as np
import os
import time
import random
from model import resnet_model
from data.CL_loader import make_dataloader
from loss import make_loss
from config import get_parser
import pickle as pkl
from eval import eval
import torch.nn.functional as F
import utils

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


def train(args, model, optimizer, criterion, task_id, gids=None, old_model=None):

    model.train()
    t0 = int(time.time())

    for epoch in range(args.num_epochs):
        train_loss = []
        dmml_losses = []
        know_distill_losses = []

        if epoch % 10 == 0:
            dataloader = make_dataloader(args, task_id, epoch)

        print('=== Epoch {}/{} ==='.format(epoch, args.num_epochs))
        adjust_lr_exp(optimizer, args.lr, epoch+1, args.num_epochs, args.lr_decay_start_epoch)
        
        for iteration, (image, label) in enumerate(dataloader):
            if args.cuda:
                image, label = image.cuda(gids[0]), label.cuda(gids[0])
            ### feat is in fact feat_new_model
            feat = model(image)

            ####### 1, dmml loss ##############
            dmml_loss = criterion(feat, label)

            ####### 2, distillation  ##############
            if task_id > 0 and args.weight_knowledge_distill > 0:
                feat_old_model=old_model(image)

                ####### with new model
                reshape_feat_new_model=feat.reshape(-1,args.num_support+args.num_query,2048)
                enc_data_query = reshape_feat_new_model[:, args.num_support:, :].squeeze(1)

                if args.distillation_dist_metric=='euclidean':
                    enc_proto = reshape_feat_new_model[:,:args.num_support,:].mean(1)
                    mix_task_new_logits = utils.decode(enc_proto, enc_data_query)
                elif args.distillation_dist_metric=='cosine':
                    enc_proto = F.normalize(reshape_feat_new_model[:,:args.num_support,:]).mean(1)
                    mix_task_new_logits = utils.cosine_decode(enc_proto, enc_data_query)
                else:
                    raise NotImplementedError
                
                if args.remove_positive_pair:
                    identity=torch.eye(len(mix_task_new_logits))
                    mix_task_new_logits = mix_task_new_logits[(1-identity).bool()]
                    mix_task_new_logits = mix_task_new_logits.reshape(len(identity),-1)

                mix_task_new_logits = F.softmax(mix_task_new_logits,dim=1)

                if args.temperature != 1.0:
                    eps=1e-5
                    T=args.temperature
                    mix_task_new_logits = F.normalize(mix_task_new_logits.pow(1/T), dim=1, p=1)
                    mix_task_new_logits = F.normalize(mix_task_new_logits + eps / mix_task_new_logits.size(1), dim=1, p=1)

                ####### with old model
                reshape_feat_old_model = feat_old_model.reshape(-1, args.num_support + args.num_query, 2048)
                enc_data_query = reshape_feat_old_model[:, args.num_support:, :].squeeze(1)
                if args.distillation_dist_metric=='euclidean':
                    enc_proto = reshape_feat_old_model[:, :args.num_support, :].mean(1)
                    mix_task_old_logits = utils.decode(enc_proto, enc_data_query)
                elif args.distillation_dist_metric=='cosine':
                    enc_proto = F.normalize(reshape_feat_old_model[:, :args.num_support, :]).mean(1)
                    mix_task_old_logits = utils.cosine_decode(enc_proto, enc_data_query)
                else:
                    raise NotImplementedError
                
                if args.remove_positive_pair:
                    identity = torch.eye(len(mix_task_old_logits))
                    mix_task_old_logits = mix_task_old_logits[(1-identity).bool()]
                    mix_task_old_logits = mix_task_old_logits.reshape(len(identity),-1)

                mix_task_old_logits = F.softmax(mix_task_old_logits,dim=1)
                
                if args.temperature != 1.0:
                    eps=1e-5
                    T=args.temperature
                    mix_task_old_logits = F.normalize(mix_task_old_logits.pow(1/T), dim=1, p=1)
                    mix_task_old_logits = F.normalize(mix_task_old_logits + eps / mix_task_old_logits.size(1), dim=1, p=1)

                kl_div_mix_task = (mix_task_old_logits.clamp(min=1e-4) * (mix_task_old_logits.clamp(min=1e-4)
                                        / mix_task_new_logits.clamp(min=1e-4)).log()).sum() / len(mix_task_old_logits)
                kl_div_mix_task= kl_div_mix_task * args.weight_knowledge_distill
            else:
                kl_div_mix_task=torch.tensor(0).cuda(gids[0])
            #################################################

            loss = dmml_loss  + kl_div_mix_task
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print training info
            train_loss.append(loss.item())
            dmml_losses.append(dmml_loss.item())
            know_distill_losses.append(kl_div_mix_task.item())

            print('Episode: {}, Loss: {:.6f}, dmml_loss: {:.6f}, kl_div_mix_task: {:.6f} '
                  .format(iteration, loss.item(), dmml_loss.item(),  kl_div_mix_task.item()))

        avg_training_loss = np.mean(train_loss)
        avg_dmml_losses = np.mean(dmml_losses)
        avg_know_distill_losses = np.mean(know_distill_losses)

        print('Average loss: {:.6f},dmml_losses: {:.6f}, know_distill_losses: {:.6f}'
              .format(avg_training_loss,avg_dmml_losses, avg_know_distill_losses))
        t = int(time.time())
        print('Time elapsed: {}h {}m'.format((t - t0) // 3600, ((t - t0) % 3600) // 60))

    model_save_path = os.path.join(args.exp_root, args.method, 
                                '{}_model_last_task_{}.pth'.format(args.method, task_id))
    if gids is not None and len(gids) > 1:
        torch.save(model.module.state_dict(), model_save_path)
    else:
        torch.save(model.state_dict(), model_save_path)
    print('Final model saved.')

    mAP, CMC=eval(gid=gids[0], dataset=args.dataset, dataset_root=args.dataset_root, which='last',
                  exp_dir=os.path.join(args.exp_root, args.method),method=args.method, task_id=task_id)

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
    if args.loss_type == 'dmml':
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

    ##############Initialization
    print('Initializing...')
    if args.cuda:
        gpus = ''.join(args.gpu.split())
        gids = [int(gid) for gid in gpus.split(',')]
    else:
        gids = None

    ############## if seed is not given, randomly generate a seed
    if args.manual_seed == None:
        args.manual_seed=int(time.time())

    args.manual_seed=int(args.manual_seed)
    init_seed(args, gids)
    print(f'seed is set to {args.manual_seed}.')
    ################################
    if not os.path.exists(os.path.join('./result', args.dataset)):
        os.makedirs(os.path.join('./result', args.dataset))

    text_file = os.path.join('./result', args.dataset, args.method + '.txt')

    f = open(text_file, 'a')
    print(args,file=f)
    f.close()
    ########### Training ###########
    if not os.path.exists( os.path.join(args.exp_root, args.method)):
        os.makedirs( os.path.join(args.exp_root, args.method))

    if args.dataset=='market1501':
        TOTAL_TASK_NUM=751
        BASE_CLS_NUM = 76
        TASK_NUM = 10
    else:
        raise NotImplementedError
        
    CLS_NUM_PER_TASK = (TOTAL_TASK_NUM - BASE_CLS_NUM) // (TASK_NUM - 1)
    args.classes_per_task = CLS_NUM_PER_TASK

    for task_id in range(args.start_task_id,TASK_NUM):
        model = make_model(args, gids)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = make_loss(args, gids)
        if task_id>0:
            ############ load checkpoint and copy old model
            model.to('cpu')
            model_save_path = os.path.join(args.exp_root, args.method,
                                               '{}_model_last_task_{}.pth'.format(args.method, task_id - 1))

            print('load ckpt from {}'.format(model_save_path))
            model.load_state_dict(torch.load(model_save_path, map_location='cpu'))

            if gids is not None:
                model = model.cuda(gids[0])
                if len(gids) > 1:
                    model = DataParallel(model, gids)

            old_model = copy.deepcopy(model)
            for name,para in old_model.named_parameters():
                para.requires_grad = False

            old_model.eval()
            print('load ckpt done!')

            if not os.path.exists(os.path.join(args.exp_root, args.method)):
                os.makedirs(os.path.join(args.exp_root, args.method))

            print(f'Starting training {task_id} ...')
            mAP, Rank_1 = train(args, model, optimizer, criterion, task_id, gids, old_model=old_model)
        else:
            if gids is not None:
                model = model.cuda(gids[0])
                if len(gids) > 1:
                    model = DataParallel(model, gids)
            mAP, Rank_1 = train(args, model, optimizer, criterion, task_id, gids)

        print('After learning TASK {}, the mAP is {:.4f} and Rank-1 is {:.4f}'.format(task_id, mAP, Rank_1))

        text_file = os.path.join('./result', args.dataset, args.method + '.txt')

        f = open(text_file, 'a')
        print('task_id {}, {:.4f}, {:.4f}'.format(task_id, mAP, Rank_1),file=f)
        print('write to file done!')
        f.close()
        print(f'Training {task_id} completed.')


if __name__ == '__main__':
    main()
