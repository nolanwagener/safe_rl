import os
import os.path as osp
from copy import deepcopy
import gym
import numpy as np
import scipy.signal
import torch.nn as nn
from mjmpc.envs import GymEnvWrapper
from mjmpc.envs.vec_env import SubprocVecEnv
from mjmpc.policies import MPCPolicy
from mjmpc.utils import helpers
from .base import Intervener

def discounted_return(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[-1]

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class HalfCheetahMpcIntervener(Intervener):
    def __init__(self, mppi_params, sim_env_name='mjmpc:half_cheetah-v0',
                 sim_env_asset_path=None, mode=Intervener.MODE.TERMINATE, intv_lim=0.01,
                 cost_smoothing=0.05, gamma=0.99, only_q=False, **kwargs):
        super().__init__(mode)
        self.cost_fn = ((lambda d: np.array(d == 0., dtype=np.float32))
                        if cost_smoothing == 0.
                        else lambda d: np.maximum(0, 1 - d/cost_smoothing))
        self.gamma = gamma
        self.intv_lim = intv_lim
        self.only_q = only_q

        if sim_env_asset_path is None:
            sim_env_asset_path = 'half_cheetah.xml'
        else:
            sim_env_asset_path = osp.join(os.getcwd(), sim_env_asset_path)
        self.sim_env = gym.make(sim_env_name, asset_path=sim_env_asset_path)
        tmp_env = GymEnvWrapper(self.sim_env)
        mppi_params['d_obs'] = tmp_env.d_obs
        mppi_params['d_state'] = tmp_env.d_state
        mppi_params['d_action'] = tmp_env.d_action
        mppi_params['action_lows'] = tmp_env.action_space.low
        mppi_params['action_highs'] = tmp_env.action_space.high

        if 'num_cpu' and 'particles_per_cpu' in mppi_params:
            mppi_params['num_particles'] = mppi_params['num_cpu']*mppi_params['particles_per_cpu']

        def make_env():
            gym_env = gym.make(sim_env_name, asset_path=sim_env_asset_path)
            rollout_env = GymEnvWrapper(gym_env)
            rollout_env.real_env_step(False)
            return rollout_env
        num_cpu = mppi_params['num_cpu']
        sim_env = SubprocVecEnv([make_env for _ in range(num_cpu)])

        def rollout_fn(num_particles, horizon, mean, noise, mode):
            obsv, rewv, actv, donev, infov, next_obsv = sim_env.rollout(num_particles,
                                                                        horizon,
                                                                        mean.copy(),
                                                                        noise,
                                                                        mode)
            sim_trajs = dict(
                observations=obsv.copy(),
                actions=actv.copy(),
                costs=-1.*rewv.copy(),
                dones=donev.copy(),
                next_observations=next_obsv.copy(),
                infos=helpers.stack_tensor_dict_list(infov.copy())
            )
            return sim_trajs

        mppi_params.pop('particles_per_cpu', None)
        mppi_params.pop('num_cpu', None)

        self.v_policy = MPCPolicy(controller_type='mppi', param_dict=mppi_params)
        self.v_policy.controller.set_sim_state_fn = sim_env.set_env_state
        self.v_policy.controller.rollout_fn = rollout_fn

        q_mppi_params = deepcopy(mppi_params)
        q_mppi_params['horizon'] -= 1
        self.q_policy = MPCPolicy(controller_type='mppi', param_dict=q_mppi_params)
        self.q_policy.controller.set_sim_state_fn = sim_env.set_env_state
        self.q_policy.controller.rollout_fn = rollout_fn

    def reset(self, **kwargs):
        self.v_policy.reset()
        self.q_policy.reset()
        self.has_intv = False

    def set_state(self, env_state):
        self.state = deepcopy(env_state)
        self.sim_env.set_env_state(env_state)

    def should_intervene(self, action):
        if self.only_q:
            self.sim_env.set_env_state(self.state)
            self.sim_env.step(action)
            self.q_policy.controller.optimize(self.sim_env.get_env_state(),
                                              hotstart=False)
            q_actions = np.concatenate([action[np.newaxis],
                                        self.q_policy.controller.mean_action])
            q = self.evaluate(q_actions)
            self.q_policy.controller._shift()
            return q > self.intv_lim

        self.v_policy.controller.optimize(self.state, hotstart=False)
        self.sim_env.set_env_state(self.state)
        self.sim_env.step(action)
        self.q_policy.controller.optimize(self.sim_env.get_env_state(), hotstart=False)
        v_actions = self.v_policy.controller.mean_action
        q_actions = np.concatenate([action[np.newaxis], self.q_policy.controller.mean_action])
        self.a_safe = np.array(v_actions[0])

        v = self.evaluate(v_actions)
        q = self.evaluate(q_actions)

        self.v_policy.controller._shift()
        self.q_policy.controller._shift()

        return q - v > self.intv_lim

    def evaluate(self, actions):
        self.sim_env.set_env_state(self.state)
        distances, v = [], 0.
        for t in range(actions.shape[0]):
            _, _, _, info = self.sim_env.step(actions[t])
            height = info['height']
            distances.append(np.maximum(0., np.minimum(height-0.4, 1.-height)))
            if distances[-1] == 0.:
                v = self.gamma**(t+1) / (1 - self.gamma)
                break
        costs = self.cost_fn(np.array(distances))
        return self.gamma*(discounted_return(costs, self.gamma) + v)

    def safe_action(self):
        if not self.has_intv: # TODO: adjust for off-policy setting
            self.has_intv = True
            return np.array(self.a_safe)
        return self.v_policy.get_action(self.state)[0]

class HalfCheetahHeuristicIntervener(Intervener):
    def __init__(self, sim_env_name='mjmpc:half_cheetah-v0', sim_env_asset_path=None,
                 min_height=0.4, max_height=1., **kwargs):
        super().__init__(Intervener.MODE.TERMINATE)
        if sim_env_asset_path is None:
            sim_env_asset_path = 'half_cheetah.xml'
        else:
            sim_env_asset_path = osp.join(os.getcwd(), sim_env_asset_path)
        self.sim_env = gym.make(sim_env_name, asset_path=sim_env_asset_path).unwrapped
        self.min_height = min_height
        self.max_height = max_height

    def reset(self, **kwargs):
        self.sim_env.reset()

    def set_state(self, env_state):
        self.sim_env.set_env_state(env_state)

    def should_intervene(self, action):
        _, _, _, info = self.sim_env.step(action)
        height = info['height']
        return not self.min_height <= height <= self.max_height
