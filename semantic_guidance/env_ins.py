import sys
import json
import torch
import numpy as np
import argparse
import torchvision.transforms as transforms
import cv2
# from DRL.ddpg import decode
import utils.util as util
from utils.dataloader import ImageDataset
import random
from Renderer.model import FCN
import pandas as pd

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Paint:
    def __init__(self, batch_size, max_eps_len, dataset='cub200', use_gbp=False, use_bilevel=False, gpu=0,
                 width=128, high_res=False, bundle_size=5):
        self.batch_size = batch_size
        self.max_eps_len = max_eps_len
        self.action_space = (13)
        self.use_gbp = use_gbp
        self.use_bilevel = use_bilevel
        self.width = width
        self.high_res = high_res
        self.bundle_size = bundle_size

        if self.high_res:
            self.width = 256

        # define and load renderer
        self.decoder = FCN(self.high_res)
        self.decoder.load_state_dict(torch.load('../data/renderer{}.pkl'.format("_256" if self.high_res else "")))
        self.decoder.cuda(gpu)

        for param in self.decoder.parameters():
            param.requires_grad = False

        # gpu and distributed parameters
        self.gpu = gpu

        # define observation space
        self.observation_space = (self.batch_size, self.width, self.width, 7 + use_gbp + use_bilevel)
        self.test = False

        # dataset name
        self.dataset = dataset
        if dataset == 'cub200':
            df = pd.read_csv('../data/cub200/CUB_200_2011/images.txt', sep=' ', index_col=0, names=['idx', 'img_names'])
            img_names = list(df['img_names'])
            self.data = np.array(
                ["../data/cub200/CUB_200_2011/images/{}.jpg".format(img_name[:-4]) for img_name in img_names])
            self.seg = np.array(
                ["../data/cub200/CUB_200_2011/segmentations_pred/{}.jpg".format(img_name[:-4]) for img_name in
                 img_names])
            self.gbp = np.array(
                ["../data/cub200/CUB_200_2011/gbp_global/{}.jpg".format(img_name[:-4]) for img_name in img_names])

            df_ = pd.read_csv('../data/cub200/CUB_200_2011/bounding_boxes_pred.txt', sep=' ', index_col=0)
            x, y, w, h = np.array(df_['x']).astype(int), np.array(df_['y']).astype(int), np.array(df_['w']).astype(int), \
                         np.array(df_['h']).astype(int)
            self.bbox = np.array(list(zip(x, y, w, h)))

            # random shuffle data
            shuffled_indices = np.arange(len(self.data)).astype(np.int32)
            np.random.shuffle(shuffled_indices)
            self.data = self.data[shuffled_indices]
            self.seg = self.seg[shuffled_indices]
            self.bbox = self.bbox[shuffled_indices]
            self.gbp = self.gbp[shuffled_indices]

    def load_data(self, num_test=2000):
        # random shuffle data
        shuffled_indices = np.arange(len(self.data)).astype(np.int32)
        np.random.shuffle(shuffled_indices)

        # divide data into train and test
        train_data, test_data = self.data[shuffled_indices[num_test:]], self.data[shuffled_indices[:num_test]]

        # divide gbp data
        if self.use_gbp:
            gbp_train, gbp_test = self.gbp[shuffled_indices[num_test:]], self.gbp[shuffled_indices[:num_test]]
        else:
            gbp_train, gbp_test = None, None

        if self.use_bilevel:
            seg_train, seg_test = self.seg[shuffled_indices[num_test:]], self.seg[
                shuffled_indices[:num_test]]
            bbox_train, bbox_test = self.bbox[shuffled_indices[num_test:]], self.bbox[
                shuffled_indices[:num_test]]
        else:
            seg_train, seg_test = None, None
            bbox_train, bbox_test = None, None

        # create train and test data
        self.train_dataset = ImageDataset(train_data, gbp_list=gbp_train, seg_list=seg_train,
                                          bbox_list=bbox_train, high_res=self.high_res)
        self.test_dataset = ImageDataset(test_data, gbp_list=gbp_test, seg_list=seg_test,
                                         bbox_list=bbox_test, high_res=self.high_res)

        # record train test split
        self.num_train, self.num_test = len(train_data), num_test

    def reset(self, test=False, begin_num=False):
        self.test = test
        # self.imgid = [0] * self.batch_size
        self.gt = torch.zeros([self.batch_size, 3, self.width, self.width], dtype=torch.uint8).cuda(self.gpu)
        if self.use_gbp:
            self.gbp_gt = torch.zeros([self.batch_size, 1, self.width, self.width], dtype=torch.uint8).cuda(self.gpu)
        if self.use_bilevel:
            self.seg_gt = torch.zeros([self.batch_size, 1, self.width, self.width], dtype=torch.uint8).cuda(self.gpu)
            self.grid = torch.zeros([self.batch_size, 2, self.width, self.width], dtype=torch.uint8).cuda(self.gpu)

        # get ground truths and corresponding idxs
        if test:
            self.imgid = (begin_num + np.arange(self.batch_size)) % self.num_test
            for i in range(self.batch_size):
                img, gbp_gt, seg_gt, grid = self.test_dataset[self.imgid[i]]
                self.gt[i] = img
                if self.use_gbp:
                    self.gbp_gt[i, :] = gbp_gt
                if self.use_bilevel:
                    self.seg_gt[i, :] = seg_gt
                    self.grid[i, :] = grid
        else:
            self.imgid = np.random.choice(np.arange(self.num_train), self.batch_size, replace=False)
            for i in range(self.batch_size):
                img, gbp_gt, seg_gt, grid = self.train_dataset[self.imgid[i]]
                self.gt[i] = img
                if self.use_gbp:
                    self.gbp_gt[i, :] = gbp_gt
                if self.use_bilevel:
                    self.seg_gt[i, :] = seg_gt
                    self.grid[i, :] = grid

        self.tot_reward = ((self.gt.float() / 255) ** 2).mean(1).mean(1).mean(1)
        self.stepnum = 0
        self.canvas = torch.zeros([self.batch_size, 3, self.width, self.width], dtype=torch.uint8).cuda(self.gpu)
        self.lastdis = self.ini_dis = self.cal_dis()
        return self.observation()

    def observation(self):
        # canvas B * 3 * width * width
        # gt B * 3 * width * width
        # T B * 1 * width * width
        ob = []
        T = torch.ones([self.batch_size, 1, self.width, self.width], dtype=torch.uint8) * self.stepnum

        # canvas, img, T
        obs_list = [self.canvas, self.gt, T.cuda(self.gpu)]

        if self.use_gbp:
            obs_list += [self.gbp_gt]
        if self.use_bilevel:
            obs_list += [self.seg_gt]
            obs_list += [self.grid]

        return torch.cat(obs_list, 1)

    def cal_trans(self, s, t):
        return (s.transpose(0, 3) * t).transpose(0, 3)

    def step(self, action):
        if self.use_bilevel:
            self.canvas = (
                        self.decode_parallel(action, self.canvas.float() / 255, seg_mask=self.seg_gt.float() / 255)[
                            0] * 255).byte()
        else:
            self.canvas = (self.decode(action, self.canvas.float() / 255)[0] * 255).byte()

        self.stepnum += 1
        ob = self.observation()
        # ob = ob[:, :7, :, :]
        done = (self.stepnum == self.max_eps_len)
        reward = self.cal_reward()  # np.array([0.] * self.batch_size)
        return ob.detach(), reward, np.array([done] * self.batch_size), None

    def cal_dis(self):
        return (((self.canvas.float() - self.gt.float()) / 255) ** 2).mean(1).mean(1).mean(1)

    def cal_reward(self):
        """L2 loss difference between canvas and ground truth"""
        dis = self.cal_dis()
        reward = (self.lastdis - dis) / (self.ini_dis + 1e-8)
        self.lastdis = dis
        return util.to_numpy(reward)

    def decode_parallel(self, x, canvas, seg_mask=None, mask=False):
        canvas, _ = self.decode(x[:, :13 * self.bundle_size], canvas, mask, 1 - seg_mask)
        canvas, _ = self.decode(x[:, 13 * self.bundle_size:], canvas, mask, seg_mask)
        return canvas, _

    def decode(self, x, canvas, mask=False, seg_mask=None):  # b * (10 + 3)
        """
        Update canvas given stroke parameters x
        :param x: stroke parameters (N,13*5)
        :param canvas: current canvas state
        :return: updated canvas with stroke drawn
        """
        # 13 stroke parameters (10 position and 3 RGB color)
        x = x.contiguous().view(-1, 10 + 3)

        # get stroke on an empty canvas given 10 positional parameters
        stroke = 1 - self.decoder(x[:, :10])
        if self.high_res is True:
            stroke = stroke.view(-1, 256, 256, 1)
        else:
            stroke = stroke.view(-1, 128, 128, 1)

        # add color to the stroke
        color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
        stroke = stroke.permute(0, 3, 1, 2)
        color_stroke = color_stroke.permute(0, 3, 1, 2)

        # draw bundle_size=5 strokes at a time (action bundle)
        if self.high_res is True:
            stroke = stroke.view(-1, self.bundle_size, 1, 256, 256)
            color_stroke = color_stroke.view(-1, self.bundle_size, 3, 256, 256)
        else:
            stroke = stroke.view(-1, self.bundle_size, 1, 128, 128)
            color_stroke = color_stroke.view(-1, self.bundle_size, 3, 128, 128)

        for i in range(self.bundle_size):
            if seg_mask is not None:
                canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i] * seg_mask
            else:
                canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]

        # also return stroke mask if required
        stroke_mask = None
        if mask:
            stroke_mask = (stroke != 0).float()  # -1, bundle_size, 1, width, width

        return canvas, stroke_mask
