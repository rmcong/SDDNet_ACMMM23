
from sys import prefix
import torch
from torch.optim import lr_scheduler, optimizer
import torch.nn as nn
import torchvision
from torchvision import transforms
from tqdm import tqdm
import torchvision.utils as vutils
from networks.sddnet import SDDNet
from networks.loss import BBCEWithLogitLoss, DiceLoss, EdgeLoss, OrthoLoss, \
DiffLoss, StyleLoss, DiffLoss_2, ZeroLoss
from utils.evaluation import MyConfuseMatrixMeter
import numpy as np
import random

from torch import Tensor
import torch.nn.functional as F

from datasets.sbu_dataset_new import SBUDataset
from datasets.transforms import Denormalize

from torch.utils.tensorboard import SummaryWriter

import configargparse
import os
import logging

from utils.visualization import colorize_classid_array

from ptflops import get_model_complexity_info
import cv2
from modelsize_estimate import modelsize
cv2.setNumThreads(0)

logger = logging.getLogger('ShadowDet')
logger.setLevel(logging.DEBUG)


def seed_all(seed=10):
    """
    https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
    """
    logger.info(f"[ Using Seed : {seed} ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def create_logdir_and_save_config(args):
    paths = {}
    paths['sw_dir'] = os.path.join(args.logdir, 'summary')
    paths['ckpt_dir'] = os.path.join(args.logdir, 'ckpt_7')
    paths['val_dir'] = os.path.join(args.logdir, 'val')
    paths['test_dir'] = os.path.join(args.logdir, 'test')
    paths['log_file'] = os.path.join(args.logdir, 'train.log')
    paths['config_file'] = os.path.join(args.logdir, 'config.txt')
    paths['arg_file'] = os.path.join(args.logdir, 'args.txt')

    #### create directories #####
    for k, v in paths.items():
        if k.endswith('dir'):
            os.makedirs(v, exist_ok=True)

    #### create summary writer #####
    sw = SummaryWriter(log_dir=paths['sw_dir'])

    #### log to both console and file ####
    str2loglevel = {'info': logging.INFO, 'debug': logging.DEBUG, 'error': logging.ERROR}
    level = str2loglevel[args.loglevel]
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # create file handler which logs even debug messages
    fh = logging.FileHandler(paths['log_file'], 'w+')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    #### print and save configs #####
    msg = 'Experiment arguments:\n ============begin==================\n'
    for arg in sorted(vars(args)):
        attr = getattr(args, arg)
        msg += '{} = {}\n'.format(arg, attr)
    msg += '=============end================'
    logger.info(msg)
    
    with open(paths['arg_file'], 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        with open(paths['config_file'], 'w') as file:
            file.write(open(args.config, 'r').read())

    return sw, paths


def create_model_and_optimizer(args):
    """
    return model, optimizer, lr_schedule, start_epoch
    """
    # create model
    model = SDDNet(backbone='efficientnet-b3',
               proj_planes=16,
               pred_planes=32,
               use_pretrained=True,
               fix_backbone=False,
               has_se=False,
               dropout_2d=0,
               normalize=True,
               mu_init=0.5,
               reweight_mode='manual')
    # macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True,
    #                                        print_per_layer_stat=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    model.cuda()
    
    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.wd, lr=args.lr)
    # lr schedule
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    start_epoch = 0

    logger.info(f'model {args.model} is created!')

    if args.ckpt is not None:
        model, optimizer, lr_schedule, start_epoch = load_ckpt(model, optimizer, lr_schedule, args.ckpt)

    return model, optimizer, lr_schedule, start_epoch


def create_loss_function(args):
    if args.loss == 'bce':
        loss_function = nn.BCEWithLogitsLoss()
    elif args.loss == 'dice':
        loss_function = DiceLoss()
    elif args.loss == 'bbce':
        loss_function = BBCEWithLogitLoss()
    else:
        raise ValueError(f'{args.loss} is not supported!')

    return loss_function


def create_dataloaders(args):
    data_roots = {'SBU_train': '/data/gyc/SBU-shadow/SBUTrain4KRecoveredSmall',
                  'SBU_test': '/data/gyc/SBU-shadow/SBU-Test_rename',
                  'UCF_test': '/data/gyc/UCF',
                  'ISTD_train': '/data/gyc/ISTD_Dataset/train',
                  'ISTD_test': '/data/gyc/ISTD_Dataset/test'}

    train_dataset = SBUDataset(data_root=data_roots[args.train_data],
                               phase='train', augmentation=False, im_size=args.train_size, normalize=False)
    
    ## set drop_last True to avoid error induced by BatchNormalization
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch,
                                               shuffle=True, num_workers=args.nworker,
                                               pin_memory=True, drop_last=True)
    
    # name, split = args.eval_data.split('_')
    # val_dataset = get_datasets(name=name,
    #                            root=os.path.join(args.data_root, name),
    #                            split=split,
    #                            transform=val_tf
    #                           )
    eval_data = args.eval_data.split('+')
    eval_loaders = {}
    for name in eval_data:
        dataset = SBUDataset(data_root=data_roots[name], phase='test', augmentation=False,
                             im_size=args.eval_size, normalize=False)
        eval_loaders[name] = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch,
                                            shuffle=False, num_workers=args.nworker,
                                            pin_memory=True)
    
    # msg = "Dataloaders are prepared!\n=============================\n"
    # msg += f"train_loader: dataset={args.train_data}, num_samples={len(train_loader.dataset)}, batch_size={train_loader.batch_size}\n"
    # msg += f"val_loader: dataset={args.eval_data}, num_samples={len(val_loader.dataset)}, batch_size={val_loader.batch_size}\n"
    # # msg += f"test_loader: dataset={args.test_data}, num_samples={len(test_loader.dataset)}, batch_size={test_loader.batch_size}\n"
    # # msg += "------------------------------\n"
    # # msg += f"load_size={args.load_size}\n"
    # msg += "============================="
    msg = "Dataloaders are prepared!"
    logger.info(msg)
    
    return train_loader, eval_loaders


def save_ckpt(model, optimizer, lr_schedule, epoch, path):
    ckpt = {'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_schedule': lr_schedule.state_dict(),
            'epoch': epoch
           }
    torch.save(ckpt, path)
    logger.info(f'checkpoint has been saved to {path}!')


def load_ckpt(model, optimizer, lr_schedule, path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    lr_schedule.load_state_dict(ckpt['lr_schedule'])
    start_epoch = ckpt['epoch'] + 1
    logger.info(f'model is loaded from {path}!')
    return model, optimizer, lr_schedule, start_epoch


def visualize_sample(images: Tensor, gt: Tensor, pred: Tensor, bi_th=0.5):
    """
    visualize single sample
    Args:
        images: [2, 3, h, w] tensor
        gt: [1, h, w] tensor, int binary mask
        pred: [1, h, w] tensor, float soft mask
    Return:
        grid: visual grid
    """
    # mean=[0.485, 0.456, 0.406]
    # std=[0.229, 0.224, 0.225]
    # # mean=[0.5, 0.5, 0.5]
    # # std=[0.5, 0.5, 0.5]
    # denorm_fn = Denormalize(mean=mean, std=std)
    # images_vis = (denorm_fn(images)*255).type(torch.uint8).cpu()
    images_vis = (images*255).type(torch.uint8).cpu()
    gt_vis = (torch.cat([gt*255]*3, dim=0)).type(torch.uint8).cpu()
    pred_vis = (torch.cat([pred*255]*3, dim=0)).type(torch.uint8).cpu()
    pred_bi_vis = (torch.cat([(pred>bi_th).float()*255]*3, dim=0)).type(torch.uint8).cpu()

    # -1: false negative, 0: correct, 1: false positive
    diff = (pred > bi_th).type(torch.int8) - gt.type(torch.int8)
    logger.debug(f"unique_ids in pred_bi_vis: {torch.unique(pred_bi_vis)}")
    logger.debug(f"unique_ids in diff: {torch.unique(diff)}")
    diff_vis, _ = colorize_classid_array(diff, alpha=1., image=None,
                        colors={-1: 'green', 0:'black', 1:'red'})
    diff_vis = diff_vis.cpu()
    grid = vutils.make_grid([images_vis, gt_vis, pred_vis, pred_bi_vis, diff_vis],
                             nrow=3, padding=0)
    return grid


@torch.no_grad()
def evaluate(model, eval_loader, bi_class_th=0.5, save_dir=None, sw=None, epoch=None, prefix=''):
    """
    run inference with given eval_loader;
    save_dir: if not None, create the folder to save test results
    """
    # logger.info('====start of evaluation====')
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # cmm = ConfuseMatrixMeter(n_class=2)
    cmm = MyConfuseMatrixMeter(n_class=2)

    for i_batch, data in enumerate(tqdm(eval_loader)):
        inp = data['train_A_input'].cuda() # (n, 2, c, h, w)
        gt = data['gt'].cuda() # (n, 1, h, w)
        # requires threshold, TODO: AUROC
        pred_soft = torch.sigmoid(F.interpolate(model(inp)['logit'], size=gt.size()[-2:], mode='bilinear'))
        # pred_soft = F.interpolate(model(inp)['logit'], size=gt.size()[-2:], mode='bilinear')
        pred = (pred_soft > bi_class_th).type(torch.int64)
        cmm.update_cm(y_pred=pred.cpu(), y_label=gt.cpu())
        # save_dir = './gyc_eval'
        # vutils.save_image(pred_soft, os.path.join(save_dir, str(i_batch)+".png"))
        
        if (save_dir is not None):
            inp = F.interpolate(inp, size=gt.size()[-2:], mode='bilinear')
            for i_image, (x, y_gt, y_pred) in enumerate(zip(inp, gt, pred_soft)):
                im_grid = visualize_sample(images=x, gt=y_gt, pred=y_pred, bi_th=bi_class_th)
                save_name = f'{i_batch*eval_loader.batch_size + i_image:05d}.png'
                save_path = os.path.join(save_dir, save_name)
                vutils.save_image(im_grid/255., save_path) # save_image() takes float [0, 1.] input

    # score_dict = cmm.get_scores()
    score_dict = cmm.get_scores_binary()
    msg = 'Scores:\n==============================================='
    for k, v in score_dict.items():
        msg += f'\n\t{prefix}.{k}:{v}'
        if sw is not None:
            sw.add_scalar(f'eval/{prefix}.{k}', v, global_step=epoch)
    msg += '\n==============================================='
    logger.info(msg)
    # logger.info('====end of evaluation====')
    return score_dict


def train(model, train_loader, loss_fn, optimizer, lr_schedule, epoch, sw, args):
    """
    one epoch scan
    """
    # logger.info(f'====start of training epoch {epoch+1}/{args.total_ep}====')
    global_step = epoch * len(train_loader)
    model.train()
    # vgg_low = nn.Sequential(*list(torchvision.models.vgg16(pretrained=True).features.children())[:7])
    # vgg_low.cuda()
    # loss added
    loss_edge = EdgeLoss()
    loss_l1 = nn.L1Loss()
    loss_l2 = nn.MSELoss()
    loss_ortho = OrthoLoss()
    loss_diff = DiffLoss_2()
    loss_style = StyleLoss()
    loss_bce = nn.BCEWithLogitsLoss()
    loss_zero = ZeroLoss()
    loss_cos = nn.CosineEmbeddingLoss()

    loss_sum = 0
    for i_batch, sample in enumerate(train_loader): # mini-batch update
        global_step += 1
        image, label = sample['train_A_input'].cuda(), sample['gt'].cuda()
        # modelsize(model=model, input=image)
        logits_shadimg, logits_shadmask, logits_noshad, \
            f_low_shad, f_high_shad, f_low_noshad, f_high_noshad, f_low_feat, f_high_feat,m,s,n = model(image)

        loss1 = loss_l1(logits_shadimg, image)
        loss2 = loss_fn(logits_shadmask, label)
        # loss3 = loss_shadimg(logits_noshad, noshad_label)
        # loss3 = loss_edge(sobel_shadmask, sobel_shadmask)
        # loss3 = loss_diff(image, logits_noshad, label)
        loss3 = loss_diff(image, logits_noshad, label, logits_shadmask)
        # loss4 = loss_style(fea_label, fea_noshad)
        loss4 = loss_cos(f_low_shad, f_high_shad, torch.ones([f_low_shad.size()[0]]).cuda())
        loss5 = loss_cos(f_low_shad, f_low_feat, torch.ones([f_low_shad.size()[0]]).cuda())
        loss6 = loss_cos(f_high_shad, f_high_feat, torch.ones([f_low_shad.size()[0]]).cuda())
        # loss7 = loss_cos(f_low_noshad, f_low_shad, torch.ones([f_low_shad.size()[0]]).cuda())
        # loss8 = loss_cos(f_high_noshad, f_high_shad, torch.ones([f_low_shad.size()[0]]).cuda())
        loss7 = loss_ortho(f_low_noshad, f_low_shad)
        loss8 = loss_ortho(f_high_noshad, f_high_shad)
        loss_mask = loss2
        loss_shadimg = 0.2*loss1
        loss_noshad = 0.2*loss3
        loss_filter = 0.2*(0.5*loss5 + 0.5*loss6 +0.01*loss7+0.01*loss8)#
        loss_total = loss_mask+ loss_shadimg  + loss_noshad + loss_filter
        # sw.add_image('img', torchvision.utils.make_grid(image[0].detach().cpu(), nrow=5, padding=20, normalize=False, pad_value=1), global_step)
        # sw.add_image('shadimg', torchvision.utils.make_grid(s[0].detach().cpu().unsqueeze(dim=1), nrow=5, padding=20, normalize=False, pad_value=1), global_step)
        # sw.add_image('shadmask', torchvision.utils.make_grid(m[0].detach().cpu().unsqueeze(dim=1), nrow=5, padding=20, normalize=False, pad_value=1), global_step)
        # sw.add_image('noshad', torchvision.utils.make_grid(n[0].detach().cpu().unsqueeze(dim=1), nrow=5, padding=20, normalize=False, pad_value=1), global_step)
        # sw.add_image('f_noshad_region', torchvision.utils.make_grid(f_noshad_region[0].detach().cpu().unsqueeze(dim=1), nrow=5, padding=20, normalize=False, pad_value=1), global_step)
        # sw.add_image('f_mask_region', torchvision.utils.make_grid(f_mask_region[0].detach().cpu().unsqueeze(dim=1), nrow=5, padding=20, normalize=False, pad_value=1), global_step)
        # sw.add_image('f_low_shad', torchvision.utils.make_grid(f_low_shad[0].detach().cpu().unsqueeze(dim=1), nrow=5, padding=20, normalize=False, pad_value=1), global_step)
        # sw.add_image('f_low_noshad', torchvision.utils.make_grid(f_low_noshad[0].detach().cpu().unsqueeze(dim=1), nrow=5, padding=20, normalize=False, pad_value=1), global_step)
        # sw.add_image('f_low_feat', torchvision.utils.make_grid(f_low_feat[0].detach().cpu().unsqueeze(dim=1), nrow=5, padding=20, normalize=False, pad_value=1), global_step)
        # sw.add_image('f_high_shad', torchvision.utils.make_grid(f_high_shad[0].detach().cpu().unsqueeze(dim=1), nrow=5, padding=20, normalize=False, pad_value=1), global_step)
        # sw.add_image('f_high_noshad', torchvision.utils.make_grid(f_high_noshad[0].detach().cpu().unsqueeze(dim=1), nrow=5, padding=20, normalize=False, pad_value=1), global_step)
        # sw.add_image('f_high_feat', torchvision.utils.make_grid(f_high_feat[0].detach().cpu().unsqueeze(dim=1), nrow=5, padding=20, normalize=False, pad_value=1), global_step)

        # 1-step SGD
        if args.acc_step == 1:
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
        else:
            loss_total.backward()
            if global_step % args.acc_step == 0:
                optimizer.step()
                optimizer.zero_grad()
        loss_sum += loss_mask.item()


        # logging
        # for name, parms in model.named_parameters():
        #     if parms.grad is None:
        #         print(name)
        # for name, parms in model.named_parameters():
        #     if parms.grad is not None:
        #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
        if global_step % args.i_print == 0:
            # info_dict = {'loss_det': loss_det.item(), 'loss_inv': loss_inv.item(), 'loss_var': loss_var.item()}
            # info_dict = {'loss_img': loss1.item(), 'loss_mask': loss2.item(), 'loss_noshad': loss3.item()}
            info_dict = {'loss_mask': loss_mask.item(), 'loss_noshad': loss_noshad.item(), 'loss_shadimg': loss_shadimg.item(), \
                         'loss_filter': loss_filter.item(), 'loss_total': loss_total}
            # info_dict = {'loss_mask': loss2.item()}
            msg = f'[batch {i_batch+1}/{len(train_loader)}, epoch {epoch+1}/{args.total_ep}]: '
            for k, v in info_dict.items():
                msg += f'{k}:{v} '
                sw.add_scalar(f'train/{k}', v, global_step=global_step)
            logger.info(msg)
            # for name, parms in model.named_parameters():
                # if parms.grad is not None:
                    # sw.add_scalar(f'data/{name}', torch.mean(parms.data), global_step=global_step)
                    # sw.add_scalar(f'grad/{name}', torch.norm(parms.grad), global_step=global_step)
        # TODO: visualization during training    
        # if global_step % args.i_vis == 0:
        #     for k, v in vis_dict.items():
        #         sw.add_image(f'train/{k}', v, global_step=global_step)   
    lr_schedule.step()

    # cumulative learning
    # model.fr.set_mu(1 - (epoch/args.total_ep)**2)
    # logger.info(f'====end of training epoch {epoch+1}/{args.total_ep}====')
    return loss_sum



def main(args):
    torch.cuda.set_device(7)
    seed_all(args.seed)
    sw, paths = create_logdir_and_save_config(args)
    model, optimizer, lr_schedule, start_epoch = create_model_and_optimizer(args)
    loss_fn = create_loss_function(args)
    train_loader, val_loader_dict = create_dataloaders(args)

    if args.action == 'test':
        for name, val_loader in val_loader_dict.items():
            _ = evaluate(model, val_loader, bi_class_th=args.prob_th,
                         save_dir=os.path.join(paths['test_dir'], name),
                         prefix=name)

    elif args.action == 'train':
        # best_score = 0 # use it to track best model over training
        # best_epoch = -1
        for name, val_loader in val_loader_dict.items():
            score_dict = evaluate(model, val_loader, bi_class_th=args.prob_th, save_dir=None,
                                    sw=sw, epoch=-1, prefix=name)
        for epoch in range(start_epoch, args.total_ep):
            loss = train(model, train_loader=train_loader, loss_fn=loss_fn,
                optimizer=optimizer, lr_schedule=lr_schedule, epoch=epoch,
                sw=sw, args=args)
            # evaluate(model, val_loader, bi_class_th=args.prob_th, save_dir=paths['val_dir'], sw=sw, epoch=epoch)
            for name, val_loader in val_loader_dict.items():
                score_dict = evaluate(model, val_loader, bi_class_th=args.prob_th, save_dir=None,
                                      sw=sw, epoch=epoch, prefix=name)
            # if score_dict['iou'] > best_score:
            #     best_score = score_dict['iou']
            #     best_epoch = epoch
            # save checkpoint
            if args.save_ckpt > 0:
                ckpt_path = os.path.join(paths['ckpt_dir'], f'ep_{epoch:03d}.ckpt')
                save_ckpt(model, optimizer, lr_schedule, epoch, path=ckpt_path)
            sw.add_scalar('train/loss', loss, global_step=epoch)
                # if epoch == best_epoch:
                #     ckpt_path = os.path.join(paths['ckpt_dir'], 'best.ckpt')
                #     save_ckpt(model, optimizer, lr_schedule, epoch, path=ckpt_path)
        # after training, record best result and run inference if the save_ckpt
        # sw.add_hparams(hparam_dict={'best_epoch': best_epoch},
        #                 metric_dict={'best_iou': best_score})
        # if args.save_ckpt > 0:
        #     # run inference
        #     model, _, _, _ = load_ckpt(model, optimizer, lr_schedule,
        #                                path=os.path.join(paths['ckpt_dir'], 'best.ckpt'))
        # evaluate and save visual results
        # for name, val_loader in val_loader_dict.items():
        #     evaluate(model, val_loader, bi_class_th=args.prob_th,
        #              save_dir=os.path.join(paths['val_dir'], name), prefix=name)

    else:
        raise ValueError(f'invalid action {args.action}')
    
    # close logger
    sw.close()
    hdls = logger.handlers[:]
    for handler in hdls:
        handler.close()
        logger.removeHandler(handler)


if __name__ == '__main__':
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    
    parser.add_argument('--action', type=str, default='train', choices=['train', 'test'],
    # parser.add_argument('--action', type=str, default='test', choices=['train', 'test'],
                        help='action, train or test')

    ## model
    parser.add_argument('--model', type=str, default='BANet.efficientnet-b3', 
                        help='architecture to be used')
    parser.add_argument('--ckpt', type=str, default=None, help='ckpt to load')
    # parser.add_argument('--ckpt', type=str, default='./ckpt/gyc_m3.ckpt', help='ckpt to load')
    # /data/gyc/new_codes/logs/ckpt_7/ep_008.ckpt

    ## optimization
    parser.add_argument('--seed', type=int, default=4, help='random seed.')
    parser.add_argument('--total_ep', type=int, default=20,  help='number of epochs for training.')
    parser.add_argument('--lr', type=float, default=5e-4,  help='initial learning rate.')
    parser.add_argument('--lr_step', type=int, default=1,  help='learning rate decay frequency (in epochs).')
    parser.add_argument('--lr_gamma', type=float, default=0.7,  help='learning rate decay factor.')
    parser.add_argument('--wd', type=float, default=1e-4,  help='weight decay.')
    parser.add_argument('--loss', type=str, default='bbce', help='loss function')
    parser.add_argument('--save_ckpt', type=int, default=1, help='>0 means save ckpt during training.')

    ## data
    parser.add_argument('--train_data', type=str, default='SBU_train', help='training dataset')
    parser.add_argument('--eval_data', type=str, default='SBU_test+UCF_test', help='training dataset')
    parser.add_argument('--train_batch', type=int, default=4, help='batch_size for train and val dataloader.')
    parser.add_argument('--eval_batch', type=int, default=1, help='batch_size for train and val dataloader.')
    parser.add_argument('--train_size', type=int, default=512, help='scale images to this size for training')
    parser.add_argument('--eval_size', type=int, default=512, help='scale images to this size for evaluation')
    parser.add_argument('--nworker', type=int, default=4, help='num_workers for train and val dataloader.')

    ## evaluation
    parser.add_argument('--prob_th', type=float, default=0.5,  help='threshold for binary classification.')

    ## logging
    parser.add_argument('--logdir', type=str, default='logs', help='directory to save logs, args, etc.')
    parser.add_argument('--loglevel', type=str, default='info', help='logging level.')
    parser.add_argument('--i_print', type=int, default=10, help='training loss display frequency in mini-batchs.')
    parser.add_argument('--acc_step', type=int, default=1, help='mini-batch step.')
    # parser.add_argument('--i_vis', type=int, default=100, help='training loss display frequency in steps.')

    # ## test
    # parser.add_argument('--test_out', type=str, default='local_test_out', help='directory to save test visuals.')
    
    args = parser.parse()
    main(args)
