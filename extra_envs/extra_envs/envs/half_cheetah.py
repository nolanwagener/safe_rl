import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from extra_envs.intervener import HalfCheetahMpcIntervener

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, min_height=0.4, max_height=1., cost_smoothing=0.,
                 asset_path='half_cheetah.xml'):
        self.min_height, self.max_height = min_height, max_height
        self.cost_smoothing = cost_smoothing
        mujoco_env.MujocoEnv.__init__(self, asset_path, 5)
        utils.EzPickle.__init__(self)
        self.reward_range = (-1., 1.) # educated guess

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_run

        # Normalization
        reward /= 5.

        height = self.get_body_com("fthigh")[-1]
        done = not (self.min_height <= height <= self.max_height or height == 0.)
        distance = max(0., min(height-self.min_height, self.max_height-height))
        cost = (float(distance == 0.)
                if self.cost_smoothing == 0.
                else max(0, 1 - distance/self.cost_smoothing))
        return ob, reward, done, dict(reward_run=reward_run, cost=cost)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def get_env_state(self):
        state = self.sim.get_state()
        qpos = state.qpos.flat.copy()
        qvel = state.qvel.flat.copy()
        state = {'qpos': qpos, 'qvel': qvel}
        return state

    def get_state(self):
        return self.get_env_state()

    def set_env_state(self, state_dict):
        qpos = state_dict['qpos'].copy()
        qvel = state_dict['qvel'].copy()
        state = self.sim.get_state()
        for i in range(self.model.nq):
            state.qpos[i] = qpos[i]
        for i in range(self.model.nv):
            state.qvel[i] = qvel[i]
        self.sim.set_state(state)
        self.sim.forward()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def evaluate_success(self, paths):
        return 0.


class HalfCheetahUnconstrainedEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)
        self.reward_range = (-1000, 1000) # TODO: reasonable values

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def get_state(self):
        return self._get_obs() # TODO: write a real function

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
