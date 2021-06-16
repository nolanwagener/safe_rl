from gym.envs.registration import register

register(
    id='Point-v0',
    entry_point='extra_envs.envs:PointEnv',
    max_episode_steps=500
)
