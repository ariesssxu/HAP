from gym.envs.registration import register

register(
    id='CraftEnv-v0',
    entry_point='craft_env.env_gym:CraftEnv',
)