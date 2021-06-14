#!/usr/bin/env python3
import cv2
import random
import numpy as np
import argparse
from DRL.evaluator import Evaluator
from utils.util import *
from utils.tensorboard import TensorBoard
import time

from DRL.ddpg import DDPG

import argparse
import torch.distributed as dist
import sys, os
import torch
import torch.multiprocessing as mp
import numpy as np

# exp = os.path.abspath('.').split('/')[-1]
# writer = TensorBoard('../train_log/{}'.format(exp))
# os.system('ln -sf ../train_log/{} ./log'.format(exp))
os.system('mkdir ./model')


def train(agent, evaluate, writer, args, gpu, distributed):
    """
    :param agent: DDPG agent
    :param evaluate:
    :param writer: tensorboard summary writer
    :return: None
    """
    ## hyperparameters
    # total timesteps for training
    train_timesteps = args.train_timesteps
    # number of parallel environments for faster sample collection
    nenv = args.nenv
    # number of episodes between validation tests
    val_interval = args.val_interval
    # maximum length of a single painting episode (number of brush strokes)
    max_eps_len = args.max_eps_len
    # number of training steps per episode
    train_steps_per_eps = args.train_steps_per_eps
    # noise factor used in training
    noise_factor = args.noise_factor

    ## display progress
    # verbose: print training progress if true
    debug = args.debug

    ## load and save directories
    # path for stored model if resuming training
    load_path = args.load_path
    # parent directory for storing trained models
    output = args.output

    ## intializations
    # get current time stamp
    time_stamp = time.time()
    # initialize training steps
    step = episode = episode_steps = 0
    # initialize total reward
    tot_reward = 0.
    # initialize state
    observation = None

    # synchronize initial models
    agent.save_model(output, episode)

    # begin training
    while step <= train_timesteps:
        # update training steps
        step += 1
        # steps within an episode (cannot be greater than max_eps_len)
        episode_steps += 1

        ## take a step in concurrent environments and store samples to the replay buffer
        # reset if it is the start of episode
        if observation is None:
            observation = agent.env.reset()
            agent.reset(observation, noise_factor)
        action = agent.select_action(observation, noise_factor=noise_factor)
        observation, reward, done, _ = agent.env.step(action)
        # store r,s,a tuple to replay memory
        agent.observe(reward, observation, done, step)

        # store progress and train if episode is done
        if episode_steps >= max_eps_len and max_eps_len:
            if step > args.warmup:
                # compute validation results
                if episode > 0 and val_interval > 0 and episode % val_interval == 0:
                    reward, dist = evaluate(agent.env, agent.select_action, debug=debug)
                    if debug and gpu == 0:
                        prRed('Step_{:07d}: mean_reward:{:.3f} mean_dist:{:.3f} var_dist:{:.3f}'.format(step - 1,
                                                                                                        np.mean(reward),
                                                                                                        np.mean(dist),
                                                                                                        np.var(dist)))
                    writer.add_scalar('validate/mean_reward', np.mean(reward), step)
                    writer.add_scalar('validate/mean_dist', np.mean(dist), step)
                    writer.add_scalar('validate/var_dist', np.var(dist), step)

                # save latest model
                if episode % 200 == 0:
                    agent.save_model(output, episode)

            # get training time and update timestamp
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            # initialize total expected value and overall value loss for the episode
            tot_Q = 0.
            tot_value_loss = 0.

            if step > args.warmup:
                # step learning rate schedule after (1e4,2e4) episodes
                # also note lr[0],lr[1] are for updating critic,actor respectively.
                if step < 10000 * max_eps_len:
                    lr = (3e-4, 1e-3)
                elif step < 20000 * max_eps_len:
                    lr = (1e-4, 3e-4)
                else:
                    lr = (3e-5, 1e-4)

                # perform several training steps after each episode
                for i in range(train_steps_per_eps):
                    # train the agent
                    Q, value_loss = agent.update_policy(lr)
                    tot_Q += Q.data.cpu().numpy()
                    tot_value_loss += value_loss.data.cpu().numpy()

                # store training performance summaries
                if gpu == 0:
                    writer.add_scalar('train/critic_lr', lr[0], step)
                    writer.add_scalar('train/actor_lr', lr[1], step)
                    writer.add_scalar('train/Q', tot_Q / train_steps_per_eps, step)
                    writer.add_scalar('train/critic_loss', tot_value_loss / train_steps_per_eps, step)

            # display training progress
            if debug and gpu == 0:
                prBlack('#{}: steps:{} interval_time:{:.2f} train_time:{:.2f}' \
                        .format(episode, step, train_time_interval, time.time() - time_stamp))

            # reset/update timestamp and episode stats
            time_stamp = time.time()
            observation = None
            episode_steps = 0
            episode += 1


def setup(args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    # limit number of threads per process
    setup_pytorch_for_mpi(args)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def cleanup():
    dist.destroy_process_group()


def setup_pytorch_for_mpi(args):
    """
    Avoid slowdowns caused by each separate process's PyTorch using
    more than its fair share of CPU resources.
    """
    print('Proc %d: Reporting original number of Torch threads as %d.' % (args.rank, torch.get_num_threads()),
          flush=True)
    if torch.get_num_threads() == 1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / args.world_size), 1)
    fair_num_threads = 1
    torch.set_num_threads(fair_num_threads)
    print('Proc %d: Reporting new number of Torch threads as %d.' % (args.rank, torch.get_num_threads()), flush=True)


def demo(gpu, args):
    # rank of the current process
    args.rank = args.nr * args.gpus + gpu
    # setup dist process
    setup(args)

    # Random seed
    seed = 10000 * gpu + gpu
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # # summary writer
    writer = TensorBoard(args.LOG_DIR)

    # setup concurrent environments
    # define agent
    agent = DDPG(args.batch_size, args.nenv, args.max_eps_len,
                 args.tau, args.discount, args.rmsize,
                 writer, args.load_path, args.output, args.dataset,
                 use_gbp=args.use_gbp, use_bilevel=args.use_bilevel,
                 gbp_coef=args.gbp_coef, seggt_coef=args.seggt_coef,
                 gpu=gpu, distributed=False,
                 high_res=args.high_res, bundle_size=args.bundle_size)
    evaluate = Evaluator(args, writer)

    # display state, action space info
    if gpu == 0:
        print('observation_space', agent.env.observation_space, 'action_space', agent.env.action_space)

    # begin training
    train(agent, evaluate, writer, args, gpu=gpu, distributed=True)

    if gpu == 0:
        print("Training finished")

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning to Paint')

    # hyper-parameter
    parser.add_argument('--dataset', type=str, default='cub200', choices=['cub200'],
                        help='dataset')
    parser.add_argument('--warmup', default=400, type=int,
                        help='timestep without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.95 ** 5, type=float, help='discount factor (gamma)')
    parser.add_argument('--batch_size', default=96, type=int, help='minibatch size')
    parser.add_argument('--bundle_size', default=5, type=int, help='action bundle size')
    parser.add_argument('--rmsize', default=800, type=int, help='replay memory size')
    parser.add_argument('--nenv', default=96, type=int,
                        help='concurrent environment number/ number of environments')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--max_eps_len', default=40, type=int, help='max length for episode (*)')
    parser.add_argument('--noise_factor', default=0, type=float, help='noise level for parameter space noise')
    parser.add_argument('--val_interval', default=50, type=int, help='episode interval for performing validation')
    parser.add_argument('--val_num_eps', default=5, type=int, help='episodes used for performing validation')
    parser.add_argument('--train_timesteps', default=int(2e6), type=int, help='total training steps')
    parser.add_argument('--train_steps_per_eps', default=10, type=int, help='number of training steps per episode')
    parser.add_argument('--load_path', default=None, type=str, help='Load model and resume training')
    parser.add_argument('--exp_suffix', default='base', type=str,
                        help='suffix for providing additional experiment info')
    parser.add_argument('--output', default='./model', type=str, help='Output path for storing model')
    parser.add_argument('--use_gbp', action='store_true', help='use gbp info along with rgb image')
    parser.add_argument('--use_bilevel', action='store_true', help='use semantic class maps info along with rgb image')
    parser.add_argument('--bundled_seggt', action='store_true',
                        help='all strokes in a bundle should belong to same class')
    parser.add_argument('--gbp_coef', default=1.0, type=float, help='coefficient for gbp reward')
    parser.add_argument('--seggt_coef', default=1.0, type=float, help='coefficient for seggt reward')
    parser.add_argument('--high_res', action='store_true', help='use high resolution gt(256 x 256)')
    parser.add_argument('--debug', dest='debug', action='store_true', help='print some info')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')

    # parse args
    args = parser.parse_args()

    # log directory
    exp_type = os.path.abspath('.').split('/')[-1]
    LOG_DIR = "../train_log/{}/".format(exp_type)
    # choose log directory based on the experiment
    exp_name = "{}/nenv{}_batchsize{}_maxstep_{}_tau{}_memsize{}_{}{}{}_{}".format(args.dataset, args.nenv,
                                                                                  args.batch_size,
                                                                                  args.max_eps_len, args.tau,
                                                                                  args.rmsize,
                                                                                  "bundlesize{}".format(args.bundle_size),
                                                                                  "_gbp{}".format(args.gbp_coef) if args.use_gbp else "",
                                                                                  "_seggt{}".format(args.seggt_coef) if args.use_bilevel else "",
                                                                                  args.exp_suffix)

    # create summary writer
    LOG_DIR += exp_name
    args.LOG_DIR = LOG_DIR

    # create output directory
    args.output = get_output_folder(args.output + "/" + exp_name, "Paint")

    # total number of processes
    args.world_size = args.gpus * args.nodes

    print("starting")
    # launch multiple processes
    mp.spawn(demo, nprocs=args.gpus, args=(args,), join=True)
