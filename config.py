import argparse
from utils import float_or_string

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset',
                        type=str,
                        default='market1501')

    parser.add_argument('--dataset_root',
                        type=str,
                        help='dataset path',
                        default=f'datasets/market1501')

    parser.add_argument('--exp_root',
                        type=str,
                        help='path to store models and logs',
                        default=f'dmml/market1501')

    parser.add_argument('--num_epochs',
                        type=int,
                        help='number of training epochs',
                        default=600)

    parser.add_argument('--lr',
                        type=float,
                        help='learning rate',
                        default=2e-4)

    parser.add_argument('--lr_decay_start_epoch',
                        type=int,
                        help='epoch from when learning rate starts to decay exponentially',
                        default=300)

    parser.add_argument('--weight_decay',
                        type=float,
                        help='L2 weight decay',
                        default=1e-4)

    parser.add_argument('--num_classes',
                        type=int,
                        help='number of classes per episode',
                        default=32)
    #############################################
    parser.add_argument('--num_support',
                        type=int,
                        help='number of support samples per class (for DMML)',
                        default=5)

    parser.add_argument('--num_query',
                        type=int,
                        help='number of query samples per class (for DMML)',
                        default=1)
    ##################################################
    parser.add_argument('--num_instances',
                        type=int, 
                        help='number of instances per class (not for DMML)',
                        default=6)

    parser.add_argument('--distance_mode',
                        type=str,
                        help='distance measurement method of DMML, \
                             which can be chosen from \'center_support\' and \'hard_mining\'',
                        default='hard_mining')

    parser.add_argument('--margin',
                        type=float_or_string,
                        help='margin parameter for contrastive loss, triplet loss or DMML loss',
                        default=0.4)

    parser.add_argument('--img_height',
                        type=int,
                        help='height of resized input images',
                        default=256)

    parser.add_argument('--img_width',
                        type=int,
                        help='width of resized input images',
                        default=128)
    ########################################################################
    parser.add_argument('--loss_type',
                        type=str,
                        default='dmml')

    parser.add_argument('--remove_downsample',
                        action='store_true',
                        help='whether to remove the final downsample operation in resnet50 model')

    parser.add_argument('--random_erasing',
                        action='store_true',
                        help='whether to use random erasing for data augmentation')

    parser.add_argument('--num_workers',
                        type=int,
                        help='number of subprocesses for data loading',
                        default=24)

    parser.add_argument('--manual_seed',
                        help='manual seed for initialization',
                        default=None,)

    parser.add_argument('--cuda',
                        action='store_true',
                        help='whether to use cuda')

    parser.add_argument('--gpu',
                        type=str,
                        help='id of gpu device(s) to be used',
                        default='0')

    ##################### some new or important parameters ##########

    parser.add_argument('--method',
                        type=str,
                        default='DwoPP')
    parser.add_argument('--model_save_name',
                        type=str,
                        default='DwoPP')
    parser.add_argument('--classes_per_task',
                        type=int,
                        default=75)
    #####################################
    parser.add_argument('--probability',
                        type=float,
                        default=0.2)

    # preprocess_data_path
    parser.add_argument('--preprocess_data_path',
                        type=str,
                        help='path to store models and logs',
                        default=f'preprocess_dataset/')
    parser.add_argument('--start_task_id',
                        type=int,
                        help='start_task_id',
                        default=0)
    ############ distill
    parser.add_argument('--weight_knowledge_distill',
                        type=float,
                        default=1.0)
    ########################################################################
    parser.add_argument('--total_classes',
                        type=int, 
                        help='total_classes',
                        default=751)
    ######################################################
    parser.add_argument('--dmml_dist_metric',
                        type=str,
                        default='euclidean',
                        choices=['cosine','euclidean'])

    parser.add_argument('--distillation_dist_metric',
                        type=str,
                        default='euclidean',
                        choices=['cosine','euclidean'])
                        
    parser.add_argument('--bnneck',
                        action='store_true',
                        help='whether to use bnneck')

    parser.add_argument('--temperature',
                        type=float,
                        default=1.0)
    parser.add_argument('--remove_positive_pair',
                        action='store_true',
                        help='whether to use positive_pair')
    return parser
