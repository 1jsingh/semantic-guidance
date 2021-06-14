from __future__ import print_function, division
import os
import torch
import numpy as np
from torchvision import transforms, utils
import cv2


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


class ImageDataset:
    """
    Dataset for loading images at run-time
    """

    def __init__(self, img_list, gbp_list=None, seg_list=None, transform=None, width=128,
                 bbox_list=None, high_res=False):
        """
        Args:
            img_list (string): list of images
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_list = img_list
        self.gbp_list = gbp_list
        self.seg_list = seg_list
        self.transform = transform
        self.width = width
        self.high_res = high_res
        self.bbox_list = bbox_list

        if self.high_res is True:
            self.width = 256

        # custom transforms
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.resize = transforms.Resize((width, width))
        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx, gbp_img=None):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # flip horizontal
        flip_horizontal = np.random.rand() > 0.5

        # read rgb image
        img_name = self.img_list[idx]
        image = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)

        # create grid
        x, y, w, h = self.bbox_list[idx]
        H, W, C = image.shape
        x, y, w, h = x / W, y / H, w / W, h / H
        if flip_horizontal:
            x = 1 - x - w
        Affine_Mat_w = [w, 0, (2 * x + w - 1)]
        Affine_Mat_h = [0, h, (2 * y + h - 1)]
        M = np.c_[Affine_Mat_w, Affine_Mat_h].T
        M = torch.tensor(M).unsqueeze(0)
        grid = torch.nn.functional.affine_grid(M, (1, 3, 128, 128))  # (1,128,128,2)
        grid = (grid + 1) / 2  # scale between 0,1
        grid = torch.tensor(grid * 255, dtype=torch.uint8).permute(0, 3, 1, 2)

        image = cv2.resize(image, (self.width, self.width))
        # image = adjust_gamma(image, gamma=1.5)
        if flip_horizontal:
            image = cv2.flip(image, 1)
        image = torch.tensor(image, dtype=torch.uint8).permute(2, 0, 1)

        # initialize gbp, seg_img image
        gbp_img, seg_img = None, None

        # read gbp image
        if self.gbp_list is not None:
            gbp_fname = self.gbp_list[idx]

            gbp_img = cv2.cvtColor(cv2.imread(gbp_fname), cv2.COLOR_BGR2GRAY)
            gbp_img = cv2.resize(gbp_img, (self.width, self.width))
            if flip_horizontal:
                gbp_img = cv2.flip(gbp_img, 1)
            gbp_img = torch.tensor(gbp_img, dtype=torch.uint8).unsqueeze(0)

        if self.seg_list is not None:
            seg_fname = self.seg_list[idx]
            seg_img = cv2.cvtColor(cv2.imread(seg_fname), cv2.COLOR_BGR2GRAY)
            seg_img = cv2.resize(seg_img, (self.width, self.width))
            if flip_horizontal:
                seg_img = cv2.flip(seg_img, 1)
            # convert to tensor
            seg_img = torch.tensor(seg_img, dtype=torch.uint8).unsqueeze(0)

        return image, gbp_img, seg_img, grid