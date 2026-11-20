"""Registration of the G1-D spec tasks (task specs from isaaclab_biman_sim on the sort_shapes workspace)."""

import gymnasium as gym


gym.register(
    id="Isaac-Spec-Task-G1D-Dex1-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "tasks.g1d_tasks.spec_tasks.env_cfg:G1DSpecEnvCfg"},
)
