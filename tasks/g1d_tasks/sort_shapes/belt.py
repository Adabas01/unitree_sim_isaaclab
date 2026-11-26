"""A conveyor belt in place of the table top: a kinematic box whose surface moves (PhysX surface velocity), so
objects lying on it are carried along while the box itself stays where it is."""
from collections.abc import Callable

from isaaclab.sim.spawners.shapes.shapes import _spawn_geom_from_prim_type
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils import configclass


@clone
def spawn_belt(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from pxr import Gf, PhysxSchema

    _spawn_geom_from_prim_type(prim_path, cfg, "Cube", {"size": 1.0}, translation, orientation, scale=cfg.size)
    prim = get_current_stage().GetPrimAtPath(prim_path)
    api = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(prim)
    api.CreateSurfaceVelocityEnabledAttr().Set(True)
    api.CreateSurfaceVelocityLocalSpaceAttr().Set(True)
    api.CreateSurfaceVelocityAttr().Set(Gf.Vec3f(*(float(v) for v in cfg.surface_velocity)))
    return prim


@configclass
class BeltCfg(CuboidCfg):
    """`rigid_props` must make the body kinematic: the surface velocity acts on a rigid body's contacts."""
    func: Callable = spawn_belt
    surface_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Velocity of the surface in the box's own frame [m/s]."""
