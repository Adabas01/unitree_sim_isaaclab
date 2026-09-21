# G1-D tasks

Each G1-D task is a separate Python package. Task-specific environment code,
documentation, and JSON configuration stay together; robot, joint, DDS, and SDK
code remains shared at repository level.

| Task | Gym ID | Documentation |
| --- | --- | --- |
| Shape sorting | `Isaac-Sort-Shapes-G1D-Dex1-Joint` | [sort_shapes/README.md](sort_shapes/README.md) |

Use the same layout for additional tasks:

```text
tasks/g1d_tasks/<task_name>/
├── __init__.py
├── README.md
├── configs/
│   └── scene.json
├── env_cfg.py
├── layout.py
├── mdp.py
└── visuals.py
```

Only add a `common/` module after two tasks actually share an implementation.
