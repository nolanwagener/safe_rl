from gym.envs.registration import register

register(
    id='Point-v0',
    entry_point='extra_envs.envs:PointEnv',
    max_episode_steps=500
)

register(
    id='HalfCheetah-v0',
    entry_point='extra_envs.envs:HalfCheetahEnv',
    max_episode_steps=1000
)

register(
    id='HalfCheetahUnconstrained-v0',
    entry_point='extra_envs.envs:HalfCheetahUnconstrainedEnv',
    max_episode_steps=1000
)
