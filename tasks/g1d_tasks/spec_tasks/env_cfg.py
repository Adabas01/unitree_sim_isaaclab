"""G1-D tasks from task specs: the sort_shapes workspace (room, table, robot, controller, cameras, DDS) with a spec's
objects and its stage rubric instead of the cubes, cylinders and bins."""

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ..sort_shapes import mdp as sort_mdp
from ..sort_shapes.env_cfg import ActionsCfg, ObservationsCfg, add_cameras_and_lights, apply_sim_settings, build_workspace
from ..sort_shapes.evaluation import load_evaluation_settings
from ..sort_shapes.layout import load_settings
from ..sort_shapes.visuals import apply_robot_materials
from . import mdp
from .evaluator import SpecEvaluator
from .recorder import SpecRecorderManagerCfg
from .spec import load_spec, object_by_role, spawn_cfg

DEFAULT_SPEC = "pick_cube_45mm"
# First spawn only (before the startup event knows the torso frame): the spec layout around a nominal torso origin,
# so that objects do not overlap; every reset then places them at their exact torso-frame poses (mdp.place_objects).
NOMINAL_TORSO_XY = (0.0, 0.0)


def build_spec_scene(settings, spec):
    scene = build_workspace(settings)
    paths = {}
    for o in spec["objects"]:
        key = mdp.object_key(o["name"])
        paths[o["name"]] = f"{{ENV_REGEX_NS}}/{key}"
        x, y = (float(v) for v in o["xy_torso"])
        setattr(scene, key, RigidObjectCfg(
            prim_path=paths[o["name"]], spawn=spawn_cfg(o),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(
                NOMINAL_TORSO_XY[0] + x, NOMINAL_TORSO_XY[1] + y,
                settings["table"]["height"] + float(o["centre_z_above_table"]) + 0.01)),
        ))
    # Contacts the rubric reads: every hand link against every object (columns in spec order), and the main
    # object against its target and the table top.
    object_paths = [paths[o["name"]] for o in spec["objects"]]
    for links in mdp.HAND_LINKS.values():
        for link in links:
            setattr(scene, f"contact_{link}", ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{link}", update_period=0.0, history_length=0,
                filter_prim_paths_expr=object_paths,
            ))
    main, target = object_by_role(spec, "main"), object_by_role(spec, "target")
    if main and target:
        scene.contact_main = ContactSensorCfg(
            prim_path=paths[main], update_period=0.0, history_length=0,
            filter_prim_paths_expr=[paths[target], "{ENV_REGEX_NS}/table_top"],
        )
    add_cameras_and_lights(scene, settings)
    return scene


@configclass
class SpecObservationsCfg:
    @configclass
    class SceneCfg(ObservationGroupCfg):
        object_poses = ObservationTermCfg(func=mdp.object_poses)
        task_progress = ObservationTermCfg(func=mdp.task_progress_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy = ObservationsCfg.PolicyCfg()
    scene_state = SceneCfg()


@configclass
class SpecEventsCfg:
    robot_materials = EventTermCfg(func=apply_robot_materials, mode="startup")
    robot_payloads = EventTermCfg(func=sort_mdp.apply_robot_payloads, mode="startup")
    torso_frame = EventTermCfg(func=mdp.cache_torso_frame, mode="startup")
    reset_robot_and_objects = EventTermCfg(func=base_mdp.reset_scene_to_default, mode="reset")
    place_objects = EventTermCfg(func=mdp.place_objects, mode="reset")
    # objects with the spec key `feed_at_s` come in during the episode (mdp.feed_objects); no effect on other specs
    feed_objects = EventTermCfg(func=mdp.feed_objects, mode="interval", interval_range_s=(0.1, 0.1))


@configclass
class SpecRewardsCfg:
    task_progress = RewardTermCfg(func=mdp.task_progress, weight=1.0)


@configclass
class SpecTerminationsCfg:
    success = TerminationTermCfg(func=SpecEvaluator)
    time_out = TerminationTermCfg(func=base_mdp.time_out, time_out=True)


@configclass
class G1DSpecEnvCfg(ManagerBasedRLEnvCfg):
    # Workspace settings: sort_shapes scene.json (room, table, robot, cameras; its cubes, cylinders and bins unused).
    sorting: dict = load_settings()
    evaluation: dict = load_evaluation_settings()
    task_spec: dict = load_spec(DEFAULT_SPEC)
    # Layout perturbation per reset (mdp.place_objects): {"kind": "Default" | "VB-POSE", "seed": int}
    perturbation: dict = {"kind": "Default", "seed": 0}
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0)
    actions = ActionsCfg()
    observations = SpecObservationsCfg()
    events = SpecEventsCfg()
    rewards = SpecRewardsCfg()
    terminations = SpecTerminationsCfg()
    recorders = SpecRecorderManagerCfg()

    def __post_init__(self):
        self.scene = build_spec_scene(self.sorting, self.task_spec)
        env_seed = self.sorting["randomization"]["fixed_seed"]
        self.seed = None if env_seed == -1 else env_seed
        apply_sim_settings(self)
