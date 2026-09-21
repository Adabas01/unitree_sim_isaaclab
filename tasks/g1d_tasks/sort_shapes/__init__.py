"""Registration for the G1-D tabletop shape-sorting task."""

import gymnasium as gym


gym.register(
    id="Isaac-Sort-Shapes-G1D-Dex1-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "tasks.g1d_tasks.sort_shapes.env_cfg:G1DSortEnvCfg"},
)
