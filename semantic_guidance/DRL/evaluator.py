import numpy as np
from utils.util import *

class Evaluator(object):

    def __init__(self, args, writer):    
        self.val_num_eps = args.val_num_eps
        self.max_eps_len = args.max_eps_len
        self.nenv = args.nenv
        self.writer = writer
        self.log = 0

    def __call__(self, env, policy, debug=False):        
        observation = None
        for episode in range(self.val_num_eps):
            # reset at the start of episode
            observation = env.reset(test=True, episode=episode)
            episode_steps = 0
            episode_reward = 0.     
            assert observation is not None            
            # start episode
            episode_reward = np.zeros(self.nenv)
            while (episode_steps < self.max_eps_len or not self.max_eps_len):
                action = policy(observation)
                observation, reward, done, (step_num) = env.step(action)
                episode_reward += reward
                episode_steps += 1
                env.save_image(self.log, episode_steps)
            dist = env.get_dist()
            self.log += 1
        return episode_reward, dist
