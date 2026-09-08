from gym.envs.registration import register
# Wall-only evaluation: avoid importing unrelated MuJoCo / d4rl environments.
register(
    id="wall",
    entry_point="env.wall.wall_env_wrapper:WallEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
