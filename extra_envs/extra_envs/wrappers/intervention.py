import numpy as np
import gym
from extra_envs.intervener.base import Intervener

class Intervention(gym.Wrapper):
    """
    Adds an intervention layer to a given Gym environment.
    The intervention can do one of the following:
        - execute a different action on the wrapped environment, with the executed action
          returned in the `info` dictionary
        - 'teleport' the agent to a different state. This also moves the time index of
          the environment forward one to several time steps.
        - move the agent to an absorbing state, and setting `done` to True
    """

    def __init__(self, env, intervener):
        """
        env: The original environment
        intervener: The intervener, of class Intervener
        """
        super().__init__(env)
        self.intervener = intervener

    def reset(self, **kwargs):
        self.intervener.reset(**kwargs)
        return self.env.reset()

    def step(self, action=None):
        """
        If action is None, we intervene.
        """
        self.intervener.set_state(self.env.get_state())

        if action is not None and not self.intervener.should_intervene(action):
            o, r, d, info = self.env.step(action)
            info['intervened'] = False
            return o, r, d, info

        nan_obs = np.full(self.observation_space.shape, float('nan'))

        if self.intervener.mode == Intervener.MODE.SAFE_ACTION:
            a_safe = self.intervener.safe_action()
            o, r, d, info = self.env.step(a_safe)
            info_intv = dict(intervened=True,
                             safe_action=a_safe,
                             safe_step_output=(o, r, d, info))
            return nan_obs, self.reward_range[0], True, info_intv

        if self.intervener.mode == Intervener.MODE.TELEPORT:
            raise NotImplementedError("Teleport not implemented")

        if self.intervener.mode == Intervener.MODE.TERMINATE:
            return nan_obs, self.reward_range[0], True, dict(intervened=True)

        raise ValueError(self.intervener.mode)
