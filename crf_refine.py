import os
import cv2
from tqdm import tqdm
from utils.evaluation import evaluate, crf_refine
from utils.misc import split_np_imgrid, get_np_imgrid
import numpy as np


# raw_dir = 'test/raw_modify3_new'
# crf_dir = 'test/crf_basic'
raw_dir = 'test/demo'
crf_dir = 'test/demo'

im_names = os.listdir(raw_dir)
os.makedirs(crf_dir, exist_ok=True)

for im_name in tqdm(im_names):
    im_grid_path = os.path.join(raw_dir, im_name)
    im_grid = cv2.imread(im_grid_path)
    # ims = split_np_imgrid(im_grid, 3, 3)
    ims = split_np_imgrid(im_grid, 3, 3)
    input_im = ims[0].copy(order='C')
    prob = ims[1][:, :, 0].copy(order='C')
    refined = crf_refine(input_im, prob)
    ims[1] = np.stack((refined,)*3, axis=2)
    im_grid_new = get_np_imgrid(np.stack(ims, axis=0), nrow=3, padding=0)
    cv2.imwrite(os.path.join(crf_dir, im_name), im_grid_new)


from utils.evaluation import evaluate

im_grid_dir = crf_dir
# pos_err, neg_err, ber, acc, df = evaluate(im_grid_dir, pred_id=1, gt_id=2, nimg=3, nrow=3)
pos_err, neg_err, ber, acc, df = evaluate(im_grid_dir, pred_id=1, gt_id=2, nimg=3, nrow=3)
print(f'\t BER: {ber:.2f}, pErr: {pos_err:.2f}, nErr: {neg_err:.2f}, acc:{acc:.4f}')