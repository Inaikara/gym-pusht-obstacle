from gymnasium.envs.registration import register

register(
    id="gym_pusht/PushT-v0",
    entry_point="gym_pusht.envs:PushTEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "state"},
)

register(
    id="gym_pusht/PushT-Obstacle",
    entry_point="gym_pusht.envs:PushTObstacleEnv",
    max_episode_steps=300,
    kwargs={"obs_type": "environment_state_agent_pos"},
)