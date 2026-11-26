"""Spawners for the sorting objects that need a mesh: the chamfered cubes with a hole in each face (mesh in
cube_mesh.py) and round cylinders (the USD cylinder shape renders as a prism with about ten sides)."""
from collections.abc import Callable
from dataclasses import MISSING

import trimesh
from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCfg
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils import configclass

from .cube_mesh import chamfered_cube


@clone
def spawn_chamfered_cube(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Rigid cube mesh; collision is its convex hull (the holes are for the cameras only)."""
    vertices, faces = chamfered_cube(cfg.size, cfg.face_flat, cfg.hole_diameter)
    stage = get_current_stage()
    _spawn_mesh_geom_from_mesh(prim_path, cfg, trimesh.Trimesh(vertices, faces, process=False),
                               translation, orientation, None, stage=stage)
    return stage.GetPrimAtPath(prim_path)


@configclass
class ChamferedCubeCfg(MeshCfg):
    func: Callable = spawn_chamfered_cube
    size: float = MISSING
    """Outer size across two opposite faces [m]."""
    face_flat: float = MISSING
    """Side of the flat square on each face [m]."""
    hole_diameter: float = 0.0
    """Diameter of the hole in the middle of each face [m]; 0 for none."""


@clone
def spawn_round_cylinder(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Rigid cylinder mesh with `sections` sides, axis along z; collision is its convex hull."""
    stage = get_current_stage()
    mesh = trimesh.creation.cylinder(radius=cfg.radius, height=cfg.height, sections=cfg.sections)
    _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation, None, stage=stage)
    return stage.GetPrimAtPath(prim_path)


@configclass
class RoundCylinderCfg(MeshCfg):
    func: Callable = spawn_round_cylinder
    radius: float = MISSING
    height: float = MISSING
    sections: int = 32
    """Sides of the prism that stands for the cylinder (2 x sections vertices: 32 keeps the convex hull within 64)."""


def dish_mesh(diameter, height, base_fraction=0.6, thickness=0.004, sections=48):
    """A shallow plate: flat round base, a rim sloping up to the full diameter at `height`; solid of revolution."""
    import numpy as np
    r, rb, t = diameter / 2, base_fraction * diameter / 2, thickness
    profile = np.array([[0.0, 0.0], [rb, 0.0], [r, height - t], [r, height], [r - 1.5 * t, height],
                        [rb - 0.5 * t, t], [0.0, t]])
    mesh = trimesh.creation.revolve(profile, sections=sections)
    mesh.apply_translation([0.0, 0.0, -height / 2])      # centred on its bounding box, like the primitives
    return mesh


@clone
def spawn_dish(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Rigid plate mesh. Collision: convex decomposition, so that objects lie inside the dish and not on a lid
    across its rim (a convex hull would close it)."""
    from pxr import UsdPhysics
    stage = get_current_stage()
    mesh = dish_mesh(cfg.diameter, cfg.height, cfg.base_fraction, cfg.thickness, cfg.sections)
    _spawn_mesh_geom_from_mesh(prim_path, cfg, mesh, translation, orientation, None, stage=stage)
    geom = stage.GetPrimAtPath(f"{prim_path}/geometry/mesh")
    if geom.IsValid() and cfg.collision_props is not None:
        UsdPhysics.MeshCollisionAPI.Apply(geom).GetApproximationAttr().Set("convexDecomposition")
    return stage.GetPrimAtPath(prim_path)


@configclass
class DishCfg(MeshCfg):
    func: Callable = spawn_dish
    diameter: float = MISSING
    height: float = MISSING
    base_fraction: float = 0.6
    """Diameter of the flat base as a fraction of the outer diameter."""
    thickness: float = 0.004
    sections: int = 48


def lump_mesh(size, seed=0, roughness=0.14):
    """An irregular rounded lump filling the box `size` (x, y, z): a sphere with bumps, scaled per axis."""
    import numpy as np
    mesh = trimesh.creation.icosphere(subdivisions=2)
    rng = np.random.default_rng(seed)
    v = mesh.vertices * (1.0 + roughness * rng.uniform(-1.0, 1.0, (len(mesh.vertices), 1)))
    v = v / np.abs(v).max(0) * (np.asarray(size, float) / 2)
    return trimesh.Trimesh(v, mesh.faces, process=False)


@clone
def spawn_lump(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Rigid lump mesh (a stand-in for a soft wad, e.g. bubble wrap or a glove); collision is its convex hull."""
    stage = get_current_stage()
    _spawn_mesh_geom_from_mesh(prim_path, cfg, lump_mesh(cfg.size, cfg.seed), translation, orientation, None, stage=stage)
    return stage.GetPrimAtPath(prim_path)


@configclass
class LumpCfg(MeshCfg):
    func: Callable = spawn_lump
    size: tuple = MISSING
    """Bounding box (x, y, z) [m]."""
    seed: int = 0
    """Seed of the bumps."""


@clone
def spawn_toy_banana(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Rigid toy banana (mesh in banana_mesh.py). Collision: convex decomposition, so that the gripper closes on the
    body and not on the filled-in curve (a convex hull is as wide as the whole curve where the hand closes). The stem
    and the tip are drawn as two coloured caps just outside the body (no collision)."""
    import numpy as np
    from pxr import Gf, UsdGeom, UsdPhysics, Vt

    from .banana_mesh import part, toy_banana
    stage = get_current_stage()
    vertices, faces, u = toy_banana(cfg.length, cfg.width, cfg.thickness)
    _spawn_mesh_geom_from_mesh(prim_path, cfg, trimesh.Trimesh(vertices, faces, process=False),
                               translation, orientation, None, stage=stage)
    geom = stage.GetPrimAtPath(f"{prim_path}/geometry/mesh")
    if geom.IsValid() and cfg.collision_props is not None:
        UsdPhysics.MeshCollisionAPI.Apply(geom).GetApproximationAttr().Set("convexDecomposition")
    for name, low, high, colour in (("stem", cfg.stem_from, 1.0, cfg.stem_color), ("tip", -1.0, -cfg.tip_from, cfg.tip_color)):
        if colour is None:
            continue
        points, triangles = part(vertices, faces, u, low, high, grow=0.0006)
        cap = UsdGeom.Mesh.Define(stage, f"{prim_path}/ends/{name}")
        cap.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in points]))
        cap.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(triangles)))
        cap.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in np.asarray(triangles).flatten()]))
        cap.CreateSubdivisionSchemeAttr("none")
        cap.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, colour))]))
    return stage.GetPrimAtPath(prim_path)


@configclass
class ToyBananaCfg(MeshCfg):
    func: Callable = spawn_toy_banana
    length: float = MISSING
    """Bounding box along the banana [m]."""
    width: float = MISSING
    """Bounding box across the curve [m] (body thickness + how far the middle bulges)."""
    thickness: float = MISSING
    """Diameter of the body [m]."""
    stem_from: float = 0.86
    """The stem cap covers the banana from this position (0 middle, 1 stem end)."""
    tip_from: float = 0.90
    stem_color: tuple | None = (0.25, 0.55, 0.12)
    tip_color: tuple | None = (0.85, 0.30, 0.12)
