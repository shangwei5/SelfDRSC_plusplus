import sys

sys.path.append('.')

import warnings

warnings.filterwarnings('ignore')

import shutil
import os
import math
import cv2
import torch
import lpips
import imageio as iio
import os.path as osp
import numpy as np
import numpy.ma as ma
from model.RIFE_sdi_recur import Model
from argparse import ArgumentParser
from tqdm import tqdm
from model.pytorch_msssim import ssim_matlab
import torch.nn.functional as F

class InputPadder:
    """ Pads images such that dimensions are divisible by divisor """
    def __init__(self, dims, divisor = 16):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_2D_grid(H, W):
    x = torch.arange(0, W, 1).float()
    y = torch.arange(0, H, 1).float()

    xx = x.repeat(H, 1)
    yy = y.view(H, 1).repeat(1, W)

    grid = torch.stack([xx, yy], dim=0)

    return grid  # (2,H,W)

def interpolate(I0, I1, num, iters, use_flip=False):
    imgs = []
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    I1 = (torch.tensor(I1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    h, w = I0.shape[-2], I0.shape[-1]
    # print(h, w)
    grid_row = generate_2D_grid(h, w)[1:].float()
    # print(grid_row.shape, grid_row)
    embts = grid_row / (h - 1)
    embts = embts.unsqueeze(0).to(I0.device)
    # sdi_maps = [torch.zeros_like(I0[:, :1, :, :]) + j / (num + 1) for j in range(1, num + 1)]
    # sdi_maps = [embts, 1 - embts]
    padder = InputPadder(I0.shape, 8)
    I0, I1 = padder.pad(I0, I1)
    sdi_maps = padder.pad(embts, 1 - embts)

    for i, sdi_map in enumerate(sdi_maps):
        if use_flip and torch.mean(sdi_map) < 0.5:
            mid = model.inference(I1, I0, sdi_map=1. - sdi_map, iters=iters)[0]
        else:
            mid = model.inference(I0, I1, sdi_map=sdi_map, iters=iters)[0]
        mid = padder.unpad(mid)
        mid = mid.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
        mid = (mid * 255.).astype(np.uint8)
        imgs.append(mid)
    return imgs


def extrapolate(I0, I1, num, iters):
    pass


def interpolate_with_anchor(I0, I1, num, iters, use_flip=False):
    """
    For non-iterative inference, we use the middle frame as anchor
    """
    imgs = []
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    I1 = (torch.tensor(I1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    # predict middle frame as anchor
    ref_sid_map = torch.zeros_like(I0[:, :1, :, :]) + 0.5
    ref_img = model.inference(I0, I1, sdi_map=ref_sid_map, iters=iters)

    sdi_maps = [torch.zeros_like(I0[:, :1, :, :]) + j / (num + 1) for j in range(1, num + 1)]

    for i, sdi_map in enumerate(sdi_maps):
        if torch.mean(sdi_map) < 0.5 and use_flip:
            mid = model.inference(I1, I0, sdi_map=1. - sdi_map, iters=iters,
                                  ref_img=ref_img, ref_sdi_map=1. - ref_sid_map)[0]
        else:
            mid = model.inference(I0, I1, sdi_map=sdi_map, iters=iters, ref_img=ref_img, ref_sdi_map=ref_sid_map)[0]
        mid = mid.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
        mid = (mid * 255.).astype(np.uint8)
        imgs.append(mid)
    return imgs


def extrapolate(I0, I1, num, iters):
    imgs = []
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    I1 = (torch.tensor(I1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    ref_sid_map = torch.ones_like(I1[:, :1, :, :])
    ref_img = I1

    sdi_maps = [torch.ones_like(I0[:, :1, :, :]) + j / (num + 1) for j in range(1, num + 1)]
    for i, sdi_map in enumerate(sdi_maps):
        mid = model.inference(I0, I1, sdi_map=sdi_map, iters=iters, ref_img=ref_img, ref_sdi_map=ref_sid_map)[0]
        mid = mid.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
        mid = (mid * 255.).astype(np.uint8)
        imgs.append(mid)
    return imgs


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--img0', type=str, default='/data1/shangwei/dataset/video/RS-GOPRO_DS/valid/GOPR0384_11_05_64X_15360fps_rs/GS/00000000_gs_000.png', help='path of start image')
    parser.add_argument('--img1', type=str, default='/data1/shangwei/dataset/video/RS-GOPRO_DS/valid/GOPR0384_11_05_64X_15360fps_rs/GS/00000000_gs_008.png', help='path of end image')
    parser.add_argument('--checkpoint', type=str, default='/home/shangwei/code/InterpAny-Clearer-main/pre-trained/RIFE/DR-RIFE-vgg/train_sdi_log',
                        help='path of checkpoint')
    parser.add_argument('--save_dir', type=str, default='/home/shangwei/code/InterpAny-Clearer-main/demo_DR-RIFE-vgg_iters/GOPR0384_11_05_64X_15360fps_rs_000/', help='where to save image results')
    # parser.add_argument('--sdi_name', type=str, default='dis_index_{}_{}_{}_avg.npy')
    # parser.add_argument('--num', type=int, nargs='+', default=[5, 5], help='number of extracted images')
    # parser.add_argument('--use_flip', action='store_true', help='according sdi value to change the order of input')
    # parser.add_argument('--anchor', action='store_true', help='use anchor for iterative inference')
    # parser.add_argument('--extra', action='store_true', help='whether to extrapolate')
    # parser.add_argument('--gif', action='store_true', help='whether to generate the corresponding gif')
    parser.add_argument('--iters', type=int, default=1, help='iteration times for inference')
    args = parser.parse_args()

    # extracted_num = 2
    # for sub_num in args.num:
    #     extracted_num += sub_num * (extracted_num - 1)

    model = Model()
    model.load_model(args.checkpoint)
    model.eval()
    model.device()

    os.makedirs(args.save_dir, exist_ok=True)
    I0 = cv2.imread(args.img0)
    I1 = cv2.imread(args.img1)
    # gif_imgs = [I0, I1]
    #
    # for sub_num in args.num:
    #     gif_imgs_temp = [gif_imgs[0], ]
    #     for i, (img_start, img_end) in enumerate(zip(gif_imgs[:-1], gif_imgs[1:])):
    #         if args.anchor:
    #             interp_imgs = interpolate_with_anchor(
    #                 img_start, img_end,
    #                 num=sub_num, iters=args.iters, use_flip=args.use_flip
    #             )
    #         else:
    #             interp_imgs = interpolate(
    #                 img_start, img_end,
    #                 num=sub_num, iters=args.iters, use_flip=args.use_flip
    #             )
    #         gif_imgs_temp += interp_imgs
    #         gif_imgs_temp += [img_end, ]
    #     gif_imgs = gif_imgs_temp
    #
    # if args.extra:
    #     gif_imgs += extrapolate(I0=gif_imgs[len(gif_imgs) // 2], I1=I1, num=extracted_num, iters=args.iters)
    #
    # print('Interpolate 2 images to {} images'.format(extracted_num))
    interp_imgs = interpolate(I0, I1, 2, iters=args.iters)

    for i, img in enumerate(interp_imgs):  #gif_imgs):
        save_path = osp.join(args.save_dir, '{:03d}.png'.format(i))
        cv2.imwrite(save_path, img)

    # if args.gif:
    #     gif_path = osp.join(args.save_dir, 'demo.gif')
    #     with iio.get_writer(gif_path, mode='I') as writer:
    #         for img in gif_imgs:
    #             writer.append_data(img[:, :, ::-1])
