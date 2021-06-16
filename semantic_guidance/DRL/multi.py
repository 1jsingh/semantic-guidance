import cv2
import torch
import numpy as np
from env_ins import Paint
import utils.util as util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class fastenv:
    def __init__(self, max_episode_length=10, nenv=64,
                 writer=None, dataset='cub200', use_gbp=False, use_bilevel=False, gpu=0, high_res=False,
                 bundle_size=5):
        self.max_episode_length = max_episode_length
        self.nenv = nenv
        self.env = Paint(self.nenv, self.max_episode_length, dataset=dataset, use_gbp=use_gbp,
                         use_bilevel=use_bilevel, gpu=gpu, high_res=high_res, bundle_size=bundle_size)
        self.env.load_data()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.writer = writer
        self.test = False
        self.log = 0
        self.gpu = gpu
        self.use_gbp = use_gbp
        self.use_bilevel = use_bilevel

    def nalignment(self, gt, canvas0):
        gt_ = (gt - self.mean) / self.std
        predictions = self.loc_model(gt_)
        M = torch.matmul(predictions, self.P)
        M = M - self.c_
        M = M.view(-1, 2, 3)
        grid = torch.nn.functional.affine_grid(M, gt.size())
        z_gt = torch.nn.functional.grid_sample(gt, grid.float())
        z_canvas0 = torch.nn.functional.grid_sample(canvas0, grid.float())
        # z_canvas1 = torch.nn.functional.grid_sample(canvas1, grid.float())
        return z_gt, z_canvas0

    def save_image(self, log, step):
        if self.gpu == 0:
            for i in range(self.nenv):
                if self.env.imgid[i] <= 10:
                    # write background images
                    canvas = util.to_numpy(self.env.canvas[i, :3].permute(1, 2, 0))
                    self.writer.add_image('{}/canvas_{}.png'.format(str(self.env.imgid[i]), str(step)), canvas, log)
            if step == self.max_episode_length:
                if self.use_bilevel:
                    # z_gt, z_canvas = self.nalignment(self.env.gt[:,:3].float() / 255,self.env.canvas[:,:3].float() / 255)
                    grid = self.env.grid[:, :2].float() / 255
                    grid = 2 * grid - 1
                    z_gt = torch.nn.functional.grid_sample(self.env.gt[:, :3].float() / 255, grid.permute(0, 2, 3, 1))
                    z_canvas = torch.nn.functional.grid_sample(self.env.canvas[:, :3].float() / 255,
                                                               grid.permute(0, 2, 3, 1))
                for i in range(self.nenv):
                    if self.env.imgid[i] < 50:
                        # write background images
                        gt = util.to_numpy(self.env.gt[i, :3].permute(1, 2, 0))
                        canvas = util.to_numpy(self.env.canvas[i, :3].permute(1, 2, 0))
                        self.writer.add_image(str(self.env.imgid[i]) + '/_target.png', gt, log)
                        self.writer.add_image(str(self.env.imgid[i]) + '/_canvas.png', canvas, log)
                        if self.use_bilevel:
                            # # also write foreground images
                            gt = util.to_numpy(z_gt[i, :3].permute(1, 2, 0))
                            canvas = util.to_numpy(z_canvas[i, :3].permute(1, 2, 0))
                            self.writer.add_image(str(self.env.imgid[i]) + '_foreground/_target.png', gt, log)
                            self.writer.add_image(str(self.env.imgid[i]) + '_foreground/_canvas.png', canvas, log)

    def step(self, action):
        with torch.no_grad():
            ob, r, d, _ = self.env.step(torch.tensor(action).cuda(self.gpu))
        return ob, r, d, _

    def get_dist(self):
        return util.to_numpy(
            (((self.env.gt[:, :3].float() - self.env.canvas[:, :3].float()) / 255) ** 2).mean(1).mean(1).mean(1))

    def reset(self, test=False, episode=0):
        self.test = test
        ob = self.env.reset(self.test, episode * self.nenv)
        return ob
