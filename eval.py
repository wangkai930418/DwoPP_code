import torch
from torchvision import transforms
import os
import random
import numpy as np
from data.market1501 import Market1501
from model import resnet_model
from utils import get_id, flip_img, compute_map

def eval(gid, dataset, dataset_root, which, exp_dir, method=None,task_id=None, verbose=False, bnneck=False):   
    mAP, CMC = main(gid=gid, dataset=dataset, dataset_root=dataset_root, \
                    which=which, exp_dir=exp_dir, verbose=verbose,method=method, \
                        task_id=task_id, bnneck=bnneck)
    return mAP, CMC

def main(gid=None, dataset=None, dataset_root=None, which=None, exp_dir=None, 
            verbose=False,method=None, task_id=None, bnneck=False):

    GPU_ID = 0                         # gpu id or 'None'
    BATCH_SIZE = 32                    # batch size when extracting query and gallery features
    
    IMG_SIZE = (256, 128)
    DATASET = 'market1501'             # market1501
    WHICH = 'last'                     # which model to load
    EXP_DIR = './dmml/market1501'
    NORMALIZE_FEATURE = True           # whether to normalize features in evaluation
    NUM_WORKERS = 8

    if gid is not None:
        GPU_ID = gid
    if dataset is not None:
        DATASET = dataset
    if which is not None:
        WHICH = which
    if exp_dir is not None:
        EXP_DIR = exp_dir

    ############################################
    if DATASET=='market1501':
        TOTAL_TASK_NUM = 751
        TASK_NUM = 10
        BASE_CLS_NUM = 76
    else:
        raise NotImplementedError
    CLS_NUM_PER_TASK = (TOTAL_TASK_NUM - BASE_CLS_NUM) // (TASK_NUM - 1)

    taskcla=[]
    for j in range(TASK_NUM):
        if j ==0:
            taskcla.append((j, BASE_CLS_NUM))
        else:
            taskcla.append((j, CLS_NUM_PER_TASK))

    print('Generating dataset...')
    ##############################################################################3
    eval_transform = transforms.Compose([transforms.Resize(IMG_SIZE, interpolation=3),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    if DATASET == 'market1501':
        datasets = {x: Market1501(dataset_root, transform=eval_transform, split=x)
                    for x in ['gallery', 'query']}
        num_classes = 751
    else:
        raise NotImplementedError

    #######################################
    remove_downsample=True
        
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=BATCH_SIZE,
               shuffle=False, num_workers=NUM_WORKERS) for x in ['gallery', 'query']}
    print('Done.')

    print('Restoring model...')

    ### You may need to modify the arguments of the model according to your training settings.

    model = resnet_model(remove_downsample=remove_downsample)
    ###########################################################################
    if method == 'joint':
        model.load_state_dict(torch.load('{}/model_{}.pth'.format(EXP_DIR, WHICH)))
    else:
        model.load_state_dict(torch.load('{}/{}_model_{}_task_{}.pth'.format(EXP_DIR, method, WHICH,task_id)))
        
    if GPU_ID is not None:
        model.cuda(GPU_ID)
    model.eval()
    print('Done.')

    #######################
    if DATASET in ['market1501']:
        print('Getting image ID...')
        gallery_cam, gallery_label = get_id(datasets['gallery'].imgs, dataset=DATASET)
        query_cam, query_label = get_id(datasets['query'].imgs, dataset=DATASET)
        print('Done.')

        # Extract feature
        print('Extracting gallery feature...')
        ##############extract_feature
        gallery_feature, _ = fast_extract_feature(model, dataloaders['gallery'],
            normalize_feature=NORMALIZE_FEATURE, gid=GPU_ID, verbose=verbose,method=method)
        print('Done.')
        print('Extracting query feature...')
        query_feature, _ = fast_extract_feature(model, dataloaders['query'],
            normalize_feature=NORMALIZE_FEATURE, gid=GPU_ID, verbose=verbose,method=method)
        print('Done.')
    #####################################
    else:
        raise NotImplementedError
    #############################
    query_cam = np.array(query_cam)
    query_label = np.array(query_label)
    gallery_cam = np.array(gallery_cam)
    gallery_label = np.array(gallery_label)

    # Evaluate
    print('Evaluating...')
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i],
            gallery_feature, gallery_label, gallery_cam)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC / len(query_label)  # average CMC
    print('Done.')
    print('Rank-1: {:.6f} Rank-5: {:.6f} Rank-10: {:.6f} mAP: {:.6f}'.format(
        CMC[0].item(), CMC[4].item(), CMC[9].item(), ap/len(query_label)))

    return ap / len(query_label), CMC


def fast_extract_feature(model, dataloaders, normalize_feature=True, gid=None, verbose=False, method=None):
    count = 0
    images_numpy = None
    with torch.no_grad():
        all_feat=[]
        all_label=[]

        for (image, label) in dataloaders:
            all_label.append(label)
            n, c, h, w = image.size()
            count += n
            if count % (10 * n) == 0 and verbose:
                print(count)
            for i in range(2):
                if i == 1:
                    image = flip_img(image.cpu())
                if gid is not None:
                    image = image.cuda(gid)

                if i==1:
                    f=model(image)
                else:
                    ff=model(image)

            fff = ff + f
            # normalize feature
            if normalize_feature:
                fnorm = torch.norm(fff, p=2, dim=1, keepdim=True)
                fff = fff.div(fnorm.expand_as(fff))

            all_feat.append(fff)

    print('total: {:d}'.format(count))

    return torch.cat(all_feat), list(torch.cat(all_label).cpu().detach().numpy())


def evaluate(qf, ql, qc, gf, gl, gc):

    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    # prediction index
    index = np.argsort(score)
    index = index[::-1]

    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    ap_tmp, CMC_tmp = compute_map(index, good_index, junk_index)
    return ap_tmp, CMC_tmp