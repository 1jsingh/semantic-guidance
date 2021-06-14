import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.distributions import Categorical
import Renderer.model as renderer
from DRL.rpm import rpm
from DRL.actor import ResNet
from DRL.critic import ResNet_wobn
from DRL.wgan import WGAN
import utils.util as util
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from DRL.multi import fastenv

# setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()


def cal_trans(s, t):
    return (s.transpose(0, 3) * t).transpose(0, 3)


class DDPG(object):
    """
    Model-based DDPG agent class
    """

    def __init__(self, batch_size=64, nenv=1, max_eps_len=40, tau=0.001, discount=0.9, rmsize=800,
                 writer=None, load_path=None, output_path=None, dataset='celeba', use_gbp=False, use_bilevel=False,
                 gbp_coef=1.0, seggt_coef=1.0, gpu=0, distributed=False, high_res=False,
                 bundle_size=5):
        # hyperparameters
        self.max_eps_len = max_eps_len
        self.nenv = nenv
        self.batch_size = batch_size
        self.gpu = gpu
        self.distributed = distributed
        self.bundle_size = bundle_size

        # set torch device
        torch.cuda.set_device(gpu)

        # gbp and seggt rewards
        self.use_gbp = use_gbp
        self.use_bilevel = use_bilevel
        self.gbp_coef = gbp_coef
        self.seggt_coef = seggt_coef

        # Multi-res and high-res
        self.high_res = high_res

        # environment
        self.env = fastenv(max_eps_len, nenv, writer, dataset, use_gbp, use_bilevel, gpu=gpu, high_res=self.high_res,
                           bundle_size=bundle_size)

        # setup local and target actor, critic networks
        # input: target, canvas, stepnum, coordconv + gbp 3 + 3 + 1 + 2
        # output: (10+3)*5 (action bundle)
        self.actor = ResNet(9 + use_gbp + use_bilevel + 2*use_bilevel, 18, 13 * bundle_size * (1 + use_bilevel), self.high_res)
        self.actor_target = ResNet(9 + use_gbp + use_bilevel + 2*use_bilevel, 18, 13 * bundle_size * (1 + use_bilevel),
                                   self.high_res)
        self.critic = ResNet_wobn(3 + 9 + use_gbp + use_bilevel + 2*use_bilevel, 18, 1,
                                  self.high_res)  # add the last canvas for better prediction
        self.critic_target = ResNet_wobn(3 + 9 + use_gbp + use_bilevel + 2*use_bilevel, 18, 1, self.high_res)

        for param in self.actor_target.parameters():
            param.requires_grad = False

        for param in self.critic_target.parameters():
            param.requires_grad = False

        # define gan
        self.wgan = WGAN(gpu, distributed, high_res=self.high_res)
        self.wgan_bg = WGAN(gpu, distributed, high_res=self.high_res)

        # transfer models to gpu
        self.choose_device()

        # optimizers for actor/critic models
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-2)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-2)

        # load actor/critic/gan models if given a load path
        if (load_path != None):
            self.load_weights(load_path)

        # set same initial weights for local and target networks
        util.hard_update(self.actor_target, self.actor)
        util.hard_update(self.critic_target, self.critic)

        # Create replay buffer (to store max rmsize hyperparameters)
        self.memory = rpm(rmsize * max_eps_len, gpu)

        # training hyper-parameters
        self.tau = tau
        self.discount = discount

        # initialize summary logs
        self.writer = writer
        self.log = 0

        # initialize state, action logs
        self.state = [None] * self.nenv  # Most recent state
        self.action = [None] * self.nenv  # Most recent action

        self.get_coord_feats()

    def get_coord_feats(self):
        # x,y coordinates for 128 x 128 or 256x256 image
        if self.high_res is True:
            coord = torch.zeros([1, 2, 256, 256])
            for i in range(256):
                for j in range(256):
                    coord[0, 0, i, j] = i / 255.
                    coord[0, 1, i, j] = j / 255.
        else:
            coord = torch.zeros([1, 2, 128, 128])
            for i in range(128):
                for j in range(128):
                    coord[0, 0, i, j] = i / 127.
                    coord[0, 1, i, j] = j / 127.

        self.coord = coord.cuda(self.gpu)

    def act(self, state, target=False):
        """
        take action for current state
        :param state: merged state (canvas,gt,t/max_eps_len,coordinates)
        :param target: bool for whether the target actor model should be used.
        :return: action (stroke parameters)
        """
        if self.high_res is True:
            state_list = [state[:, :6].float() / 255, state[:, 6:7].float() / self.max_eps_len,
                          self.coord.expand(state.shape[0], 2, 256, 256)]
        else:
            state_list = [state[:, :6].float() / 255, state[:, 6:7].float() / self.max_eps_len,
                          self.coord.expand(state.shape[0], 2, 128, 128)]

        if self.use_gbp:
            state_list += [state[:, 7:8].float() / 255.]
        if self.use_bilevel:
            state_list += [state[:, 7 + self.use_gbp: 7 + self.use_gbp + 1 + 2].float() / 255.]
            # state_list += [state[:, 7 + self.use_gbp + 1: 7 + self.use_gbp + 1].float() / 255.]
        # define merged state (canvas,gt,t/max_eps_len,coordinates)
        state = torch.cat(state_list, 1)

        if target:
            return self.actor_target(state)
        else:
            return self.actor(state)

    def update_gan(self, state):
        """
        update WGAN based on current state (first 3 channels for canvas and last three for groundtruth)
        :param state:
        :return: None
        """
        # get canvas and groundtruth from the state
        canvas = state[:, :3].float() / 255
        gt = state[:, 3: 6].float() / 255

        # update gan based on canvas and background groundtruth images
        fake, real, penal = self.wgan_bg.update(canvas, gt)

        if self.use_bilevel:
            seggt = state[:, 7 + self.use_gbp: 7 + self.use_gbp + 1].float() / 255.
            grid = state[:, 7 + self.use_gbp + 1: 7 + self.use_gbp + 1 + 2].float() / 255.
            grid = 2 * grid - 1
            canvas = torch.nn.functional.grid_sample(canvas * seggt, grid.permute(0, 2, 3, 1))
            gt = torch.nn.functional.grid_sample(gt * seggt, grid.permute(0, 2, 3, 1))

        # update gan based on canvas and groundtruth images
        fake, real, penal = self.wgan.update(canvas, gt)


    def evaluate(self, state, action, target=False):
        """
        compute model performance (rewards) for given state, action (used for both training and testing
        based on whether target or local network is used)
        :param state: combined state (canvas,ground-truth)
        :param action: stroke parameters (10 for position and 3 for color)
        :param target: bool for whether the target critic model should be used.
        :return: critic value + gan reward (used for training actor in model-based DDPG), gan reward
        """

        # get canvas, ground-truth, time from merged state (gt,canvas,t)
        gt = state[:, 3: 6].float() / 255
        canvas0 = state[:, :3].float() / 255
        T = state[:, 6: 7]

        if self.use_gbp:
            gbpgt = state[:, 7:8].float() / 255.
        if self.use_bilevel:
            seggt = state[:, 7 + self.use_gbp: 7 + self.use_gbp + 1].float() / 255.
            grid = state[:, 7 + self.use_gbp + 1: 7 + self.use_gbp + 1 + 2].float() / 255.
            grid_ = 2 * grid - 1

        # update canvas given current action
        if self.use_bilevel:
            canvas1, stroke_masks = self.env.env.decode_parallel(action, canvas0,
                                                                 mask=self.use_gbp or self.use_bilevel,
                                                                 seg_mask=seggt)
        else:
            canvas1, stroke_masks = self.env.env.decode(action, canvas0, mask=self.use_gbp or self.use_bilevel)

        # if self.use_bilevel:
        #     # canvas1 = canvas1 * seggt
        #     # canvas0 = canvas0 * seggt

        # compute bg gan reward based on difference between wgan distances (L_t - L_t-1)
        bg_reward = self.wgan_bg.cal_reward(canvas1, gt) - self.wgan_bg.cal_reward(canvas0, gt)
        bg_reward = bg_reward.view(-1)
        # gan_reward = ((canvas0 - gt) ** 2).mean(1).mean(1).mean(1) - ((canvas1 - gt) ** 2).mean(1).mean(1).mean(1)

        if self.use_gbp:
            gbp_reward = (((canvas0 - gt) * gbpgt) ** 2).mean(1).sum(1).sum(1) \
                         - (((canvas1 - gt) * gbpgt) ** 2).mean(1).sum(1).sum(1)

            gbp_reward = gbp_reward / torch.sum(gbpgt, dim=(1, 2, 3))
            gbp_reward = 1e3 * gbp_reward
        else:
            gbp_reward = torch.tensor(0.)

        if self.use_bilevel:
            canvas1_ = torch.nn.functional.grid_sample(canvas1 * seggt, grid_.permute(0, 2, 3, 1))
            canvas0_ = torch.nn.functional.grid_sample(canvas0 * seggt, grid_.permute(0, 2, 3, 1))
            # gt_, canvas0_, canvas1_ = self.nalignment(gt,canvas0_,canvas1_)
            gt_ = torch.nn.functional.grid_sample(gt * seggt, grid_.permute(0, 2, 3, 1))

        foreground_reward = self.wgan.cal_reward(canvas1_, gt_) - self.wgan.cal_reward(canvas0_, gt_)
        # foreground_reward = ((canvas0_ - gt_) ** 2).mean(1).mean(1).mean(1) - ((canvas1_ - gt_) ** 2).mean(1).mean(1).mean(1)
        foreground_reward = 2e0 * foreground_reward.view(-1)

        # total reward
        total_reward = bg_reward + gbp_reward + foreground_reward

        # get new merged state
        if self.high_res is True:
            coord_ = self.coord.expand(state.shape[0], 2, 256, 256)
        else:
            coord_ = self.coord.expand(state.shape[0], 2, 128, 128)
        state_list = [canvas0, canvas1, gt, (T + 1).float() / self.max_eps_len, coord_]

        if self.use_gbp:
            state_list += [state[:, 7:8].float() / 255.]
        if self.use_bilevel:
            state_list += [seggt]
            state_list += [grid]

        # compute merged state
        merged_state = torch.cat(state_list, 1)
        # canvas0 is not necessarily added

        if target:
            # compute Q from target network
            Q = self.critic_target(merged_state)
        else:
            # compute Q from local network
            Q = self.critic(merged_state)
        return (Q + total_reward), total_reward

    def update_policy(self, lr):
        """
        update actor, critic using current replay memory buffer and given learning rate
        :param lr: learning rate
        :return: negative policy loss (current expected reward), value loss
        """
        self.log += 1

        # set different learning rate for actor and critic
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = lr[0]
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr[1]

        # sample a batch from the replay buffer
        state, action, reward, next_state, terminal = self.memory.sample_batch(self.batch_size, device)

        # update gan model
        self.update_gan(next_state)

        # Q-learning: Q(s,a) = r(s,a) + gamma * Q(s',a')
        with torch.no_grad():
            next_action = self.act(next_state, True)
            target_q, _ = self.evaluate(next_state, next_action, True)
            target_q = self.discount * ((1 - terminal.float()).view(-1, 1)) * target_q

        # add r(s,a) to Q(s,a)
        cur_q, step_reward = self.evaluate(state, action)
        target_q += step_reward.detach()

        # critic loss and update
        value_loss = criterion(cur_q, target_q)
        self.critic.zero_grad()
        value_loss.backward(retain_graph=True)
        self.critic_optim.step()

        # actor loss and update
        action = self.act(state)
        pre_q, _ = self.evaluate(state.detach(), action)
        policy_loss = -pre_q.mean()
        self.actor.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.actor_optim.step()

        # Soft-update target networks for both actor and critic
        util.soft_update(self.actor_target, self.actor, self.tau)
        util.soft_update(self.critic_target, self.critic, self.tau)

        return -policy_loss, value_loss

    def observe(self, reward, state, done, step):
        """
        Store observed sample in replay buffer
        :param reward:
        :param state:
        :param done:
        :param step: step count within an episode
        :return: None
        """
        s0 = self.state.clone().detach().cpu()  # torch.tensor(self.state, device='cpu')
        a = util.to_tensor(self.action, "cpu")
        r = util.to_tensor(reward, "cpu")
        s1 = state.clone().detach().cpu()  # torch.tensor(state, device='cpu')
        d = util.to_tensor(done.astype('float32'), "cpu")
        for i in range(self.nenv):
            self.memory.append([s0[i], a[i], r[i], s1[i], d[i]])
        self.state = state

    def noise_action(self, noise_factor, state, action):
        """
        Add gaussian noise to continuous actions (stroke params) with zero mean and self.noise_level[i] variance
        :param noise_factor:
        :param state:
        :param action:
        :return: action (stroke params) clipped between 0,1
        """
        noise = np.zeros(action.shape)
        for i in range(self.nenv):
            action[i] = action[i] + np.random.normal(0, self.noise_level[i], action.shape[1:]).astype('float32')
        return np.clip(action.astype('float32'), 0, 1)

    def select_action(self, state, return_fix=False, noise_factor=0):
        """
        compute action given a state and noise_factor
        :param state:
        :param return_fix:
        :param noise_factor:
        :return:
        """
        self.eval()
        # compute action
        with torch.no_grad():
            action = self.act(state)
            action = util.to_numpy(action)
        # add noise to action
        if noise_factor > 0:
            action = self.noise_action(noise_factor, state, action)
        self.train()

        self.action = action
        if return_fix:
            return action
        return self.action

    def reset(self, obs, factor):
        self.state = obs
        self.noise_level = np.random.uniform(0, factor, self.nenv)

    def decode(self, x, canvas, mask=False):  # b * (10 + 3)
        """
        Update canvas given stroke parameters x
        :param x: stroke parameters
        :param canvas: current canvas state
        :return: updated canvas with stroke drawn
        """
        # 13 stroke parameters (10 position and 3 RGB color)
        x = x.view(-1, 10 + 3)

        # get stroke on an empty canvas given 10 positional parameters
        stroke = 1 - self.decoder(x[:, :10])
        stroke = stroke.view(-1, 128, 128, 1)

        # add color to the stroke
        color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
        stroke = stroke.permute(0, 3, 1, 2)
        color_stroke = color_stroke.permute(0, 3, 1, 2)

        # draw bundle_size=5 strokes at a time (action bundle)
        stroke = stroke.view(-1, self.bundle_size, 1, 128, 128)
        color_stroke = color_stroke.view(-1, self.bundle_size, 3, 128, 128)
        for i in range(self.bundle_size):
            canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]

        # also return stroke mask if required
        stroke_mask = None
        if mask:
            stroke_mask = (stroke != 0).float()  # -1, bundle_size, 1, width, width

        return canvas, stroke_mask

    def load_weights(self, path, map_location=None, num_episodes=0):
        """
        load actor,critic,gan from given paths
        :param path:
        :return:
        """
        if map_location is None:
            map_location = {'cuda:0': 'cuda:{}'.format(self.gpu)}

        if path is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor_{:05}.pkl'.format(path, num_episodes), map_location=map_location))
        self.critic.load_state_dict(
            torch.load('{}/critic_{:05}.pkl'.format(path, num_episodes), map_location=map_location))
        self.wgan.load_gan(path, map_location, num_episodes)
        self.wgan_bg.load_gan(path, map_location, num_episodes + 1)

    def save_model(self, path, num_episodes):
        """
        save trained actor,critic,gan models
        :param path: save parent dir
        :return: None
        """
        if self.gpu == 0:
            torch.save(self.actor.state_dict(), "{}/actor_{:05}.pkl".format(path, num_episodes))
            torch.save(self.critic.state_dict(), '{}/critic_{:05}.pkl'.format(path, num_episodes))
            self.wgan.save_gan(path, num_episodes)
            self.wgan_bg.save_gan(path, num_episodes + 1)

        # Use a barrier() to make sure that process 1 loads the model after process
        # 0 saves it.
        dist.barrier()
        # configure map_location properly
        device_pairs = [(0, self.gpu)]
        map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}
        self.load_weights(path, map_location, num_episodes)
        dist.barrier()
        print("done saving on cuda:{}".format(self.gpu))

    def eval(self):
        """
        set actor, critic in eval mode
        :return: None
        """
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def train(self):
        """
        set actor, critic in train mode
        :return: None
        """
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def choose_device(self):
        """
        transfer renderer, actor, critic to device
        :return: None
        """
        self.actor.cuda(self.gpu)
        self.actor_target.cuda(self.gpu)
        self.critic.cuda(self.gpu)
        self.critic_target.cuda(self.gpu)

        if self.distributed:
            self.actor = DDP(self.actor, device_ids=[self.gpu])
            self.critic = DDP(self.critic, device_ids=[self.gpu])
