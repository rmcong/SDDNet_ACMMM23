import torchvision
from networks.sddnet import SDDNet
from datasets.sbu_dataset_new import SBUDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch.nn.functional as F

from PIL import Image
import numpy as np
from torchvision import transforms
import cv2
import time
cv2.setNumThreads(0)

# ckpt_path = 'ckpt/istd_epoch_010.ckpt'
# ckpt_path = './ckpt/ucf.ckpt'
ckpt_path = './ckpt/sbu.ckpt'
# ckpt_path = './ckpt/istd.ckpt'
# ckpt_path = '/data/gyc/new_codes/logs/ckpt_7/ep_019.ckpt'
# data_root = '/data/gyc/ISTD_Dataset/test'
data_root = '/data/gyc/SBU-shadow/SBU-Test_rename'
# data_root = '/data/gyc/UCF'
# save_dir = 'test/raw_modify3_new'
save_dir = 'test/demo'
torch.cuda.set_device(7)

os.makedirs(save_dir, exist_ok=True)

model = SDDNet(backbone='efficientnet-b3',
               proj_planes=16,
               pred_planes=32,
               use_pretrained=True,
               fix_backbone=False,
               has_se=False,
               dropout_2d=0,
               normalize=True,
               mu_init=0.4,
               reweight_mode='manual')
# ckpt = torch.load(ckpt_path)
ckpt = torch.load(ckpt_path, map_location={'cuda:0': 'cuda:7'})
for i in ckpt:
    print(i)
model.load_state_dict(ckpt['model'])
# model.fr.set_mu(0.4)
model.cuda()
model.eval()


test_dataset = SBUDataset(data_root=data_root,
                    augmentation=True,
                    phase='test',
                    normalize=False, 
                    im_size=512)
# print(test_dataset[0])
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4)

with torch.no_grad():
    # img_list = [img_name for img_name in os.listdir(os.path.join(data_root, 'train_A')) if
    #                 img_name.endswith('.png')]
    # for idx, im_name in enumerate(img_list):
    #     image = Image.open(os.path.join(data_root, 'train_A', im_name))
    #     w, h = image.size
    #     img_var = Variable(img_transform(img).unsqueeze(0)).cuda()
    #     gt = Image.open(os.path.join(data_root, 'train_B', im_name))
    time_all = []
    for data in tqdm(test_loader):
        image = data['train_A_input'].cuda()
        im_name = data['im_name'][0]
        save_path = os.path.join(save_dir, im_name)
        gt = data['gt'][0]
        start = time.time()
        ans = model(image)
        end = time.time()
        time_all.append(end-start)

        img = Image.open(os.path.join(data_root, 'train_A', im_name))
        w,h = img.size
        image = transforms.Resize((h, w))(image)
        pred = transforms.Resize((h, w))(torch.sigmoid(ans['logit'].cpu())[0])
        gt = transforms.Resize((h, w))(gt)
        imgrid = torchvision.utils.save_image([image.cpu()[0], pred.expand_as(image[0]),gt.expand_as(image[0])], fp=save_path, nrow=3, padding=0)


        # pred = torch.sigmoid(ans['logit'].cpu())[0]
        # pred = (pred > 0.5).type(torch.int64)

        # noshad = ans['noshad'].cpu()[0]
        # shad = ans['shad'].cpu()[0]
        # maskimg = ans['maskimg'].cpu()[0]

        # imgrid = torchvision.utils.save_image([image.cpu()[0], pred.expand_as(image[0]),
        #  gt.expand_as(image[0]), noshad, shad, maskimg], fp=save_path, nrow=3, padding=0)
    print('average time:', np.mean(time_all) / 1)
    print('average fps:',1 / np.mean(time_all))

    print('fastest time:', min(time_all) / 1)
    print('fastest fps:',1 / min(time_all))

    print('slowest time:', max(time_all) / 1)
    print('slowest fps:',1 / max(time_all))



from utils.evaluation import evaluate

# im_grid_dir = 'test/raw'
im_grid_dir = save_dir
pos_err, neg_err, ber, acc, df = evaluate(im_grid_dir, pred_id=1, gt_id=2, nimg=3, nrow=3)
print(f'\t BER: {ber:.2f}, pErr: {pos_err:.2f}, nErr: {neg_err:.2f}, acc:{acc:.4f}')