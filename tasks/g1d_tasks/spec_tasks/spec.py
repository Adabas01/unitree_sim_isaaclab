"""Task specs from ponimatkin's isaaclab_biman_sim, loaded onto this repo's G1-D workspace.

A spec (YAML) names a task type, a prompt, the success rubric and the objects. Each object is placed in the
robot's TORSO frame, as there: `xy_torso` (x forward, y left, metres) and `orientation_torso_xyzw` in the level
frame at the torso origin, and `centre_z_above_table` above the table top. Copies of the specs live in specs/<suite>/,
each with its source commit in a header comment.

Objects supported here: primitives (Cube, Cylinder, Sphere), generated open boxes, and registered BEHAVIOR-1K objects
(`source: behavior1k`) through ponimatkin's chiral asset set (formerly isaaclab_biman_sim): its
assets/objects/registry.yaml maps `category/model` to a single-rigid-body wrapper USD around a SimReady / Palatial /
ForgeCAD asset, spawned as chiral's biman/assets.py does (scaled per axis to the spec's bounding box, mass from the
registry or USD_DENSITY over the collider volume). The asset root is outside git ($G1D_CHIRAL_ASSETS, default
~/adam_unitree/sim_assets/chiral_554fb5e; tools/fetch_spec_assets.py downloads the originals). Objects without a
loadable asset (not downloaded, not on Hugging Face, articulated) need `placeholders=True`: a stand-in of the same
box as chiral's (open box for containers, cylinder for plates and round objects, box otherwise), flagged in every
record.
"""

import functools
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import yaml

SPEC_DIR = Path(__file__).parent / "specs"
STAGES = yaml.safe_load((Path(__file__).parent / "stages.yaml").read_text())
GENERATED_USD_DIR = Path(__file__).resolve().parents[3] / "tmp" / "generated_usd"

# isaaclab_biman_sim g1sim/assets.py (commit 4d0554c)
DENSITY = 450.0                 # kg/m^3, primitive stand-ins (a 45 mm cube: 41 g)
OPEN_BOX_MASS = 0.25            # kg
OPEN_BOX_WALL = 0.006           # m
STATIC_FRICTION, DYNAMIC_FRICTION = 1.0, 0.8
CONTAINERS = {"klt", "tray", "bowl", "box", "storage_box", "toy_box", "lunch_box", "wicker_basket", "shopping_basket",
              "pencil_holder", "vase", "goblet", "desk_organizer", "toolbox", "toaster_oven", "top_cabinet",
              "test_tube_rack", "cup_holder", "beaker", "water_glass", "wineglass", "pitcher", "coffee_cup", "mug",
              "paper_cup", "realm_usd_bottle"}
# chiral biman/assets.py (commit 554fb5e): stand-in shapes and the density of registered assets
FLAT = {"plate"}
ROUND = {"apple", "lemon", "orange", "plum", "egg", "muffin", "soccer_ball", "half_apple", "can_of_soda", "cork",
         "bottle_of_water", "bottle_of_wine", "water_bottle", "glue_stick", "battery", "jar_of_puree", "marker",
         "pen", "pencil", "test_tube", "dahlia_flower"}
USD_DENSITY = 500.0             # kg/m^3 over a registered asset's collider volume (a 7 cm apple: ~90 g)
CHIRAL_ROOT = Path(os.environ.get("G1D_CHIRAL_ASSETS", "~/adam_unitree/sim_assets/chiral_554fb5e")).expanduser()
OBJECT_KEYS = ("name", "role", "asset", "bounding_box", "xy_torso", "orientation_torso_xyzw", "centre_z_above_table")


def find_spec(name_or_path):
    """A spec file: a path, or a task name under specs/ (`stack_rgy_blocks` or `UNIFOLM/stack_rgy_blocks`)."""
    path = Path(name_or_path).expanduser()
    if path.is_file():
        return path
    matches = sorted(SPEC_DIR.glob(f"{name_or_path}.yaml")) or sorted(SPEC_DIR.glob(f"*/{name_or_path}.yaml"))
    if len(matches) != 1:
        known = ", ".join(sorted(p.stem for p in SPEC_DIR.glob("*/*.yaml")))
        raise FileNotFoundError(f"No unique task spec {name_or_path!r}; known: {known}")
    return matches[0]


def load_spec(name_or_path, placeholders=False):
    """The validated spec, with `spec_file`, `spec_hash` and per object `shape` (cube, cylinder, sphere, open_box)
    and `placeholder` (True: an open box standing in for a registered container asset)."""
    path = find_spec(name_or_path)
    text = path.read_text()
    spec = yaml.safe_load(text)
    for key in ("name", "task_type", "prompt", "objects"):
        if key not in spec:
            raise ValueError(f"{path}: missing {key!r}")
    if spec["task_type"] not in STAGES:
        raise ValueError(f"{path}: unknown task_type {spec['task_type']!r}")
    names = [o.get("name") for o in spec["objects"]]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: object names must be unique")
    for o in spec["objects"]:
        missing = [key for key in OBJECT_KEYS if key not in o]
        if missing:
            raise ValueError(f"{path}: object {o.get('name')!r} lacks {missing}")
        if not o["name"].isidentifier():
            raise ValueError(f"{path}: object name {o['name']!r} must be a Python identifier")
        o["shape"], o["placeholder"] = _shape(path, o, placeholders)
    spec["spec_file"] = str(path)
    spec["spec_hash"] = hashlib.sha256(text.encode()).hexdigest()[:16]
    return spec


@functools.lru_cache(maxsize=1)
def registry():
    """chiral's assets/objects/registry.yaml ({} without an asset root)."""
    path = CHIRAL_ROOT / "assets" / "objects" / "registry.yaml"
    return yaml.safe_load(path.read_text()) or {} if path.is_file() else {}


def asset_problem(entry):
    """Why a registry entry cannot be spawned here (None: it can)."""
    if entry is None:
        return f"not in the chiral registry ({CHIRAL_ROOT})"
    if entry.get("articulated"):
        return "articulated (not supported here)"
    if not entry.get("native_extent"):
        return "registry entry without native_extent"
    wrapper = CHIRAL_ROOT / entry["usd"]
    if not wrapper.is_file():
        return f"wrapper {wrapper} missing"
    missing = [ref for ref in re.findall(r"@([^@]+)@", wrapper.read_text(errors="replace"))
               if not (wrapper.parent / ref).exists()]
    return f"not downloaded ({missing[0]})" if missing else None


def _shape(path, o, placeholders):
    asset = o["asset"]
    if asset["source"] == "primitive":
        shape = asset.get("primitive_type", "Cube").lower()
        if shape not in ("cube", "cylinder", "sphere"):
            raise ValueError(f"{path}: object {o['name']!r}: primitive_type {shape!r} not supported")
        return shape, False
    if asset["source"] == "generated":
        # keys of our own (not in chiral's specs): wall [m], mass [kg], front_height [m] (a lower wall at +x)
        if asset.get("kind", "open_box") != "open_box" or asset.get("opening"):
            raise ValueError(f"{path}: object {o['name']!r}: only plain generated open boxes are supported")
        return "open_box", False
    key = f"{asset.get('category')}/{asset.get('model')}" if asset["source"] == "behavior1k" else None
    entry = registry().get(key) if key else None
    problem = asset_problem(entry) if key else f"asset source {asset['source']!r} not supported"
    o["asset_key"] = key
    if problem is None:
        o["usd"] = str((CHIRAL_ROOT / entry["usd"]).resolve())
        o["scale"] = [float(b) / float(n) for b, n in zip(o["bounding_box"], entry["native_extent"])]
        o["mass"] = entry.get("mass")
        o["density"] = entry.get("density", USD_DENSITY)
        return "usd", False
    o["asset_problem"] = problem
    if placeholders:
        category = asset.get("category")
        bbox = [float(v) for v in o["bounding_box"]]
        if category in CONTAINERS:
            return "open_box", True
        if category in FLAT or (category in ROUND and abs(bbox[0] - bbox[1]) < 0.3 * max(bbox[:2])):
            return "cylinder", True
        return "cube", True
    raise ValueError(
        f"{path}: object {o['name']!r} ({key}): {problem}. Fetch it with tools/fetch_spec_assets.py {path.stem}, "
        "or run with --spec-placeholders (a stand-in of its box)")


def stages(spec):
    return list(STAGES[spec["task_type"]])


def object_by_role(spec, role):
    return next((o["name"] for o in spec["objects"] if o["role"] == role), None)


def local_points(o):
    """Points of the object's geometry in its own frame, for axis-aligned boxes: box corners, cylinder rims; a
    sphere is its centre ± radius on each axis (exact for any orientation, see aabb)."""
    sx, sy, sz = (float(v) / 2 for v in o["bounding_box"])
    if o["shape"] in ("cube", "open_box", "usd"):
        return np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
    if o["shape"] == "cylinder":
        r = min(sx, sy)
        ring = np.array([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]) * r
        return np.concatenate([np.c_[ring, np.full(len(ring), z)] for z in (-sz, sz)])
    return np.zeros((1, 3))


def usd_local_points(prim_path):
    """A spawned USD object's real geometry as points in its root frame, spawn scale applied: mesh vertices
    (subsampled), cube corners, cylinder / sphere rims. Port of chiral biman/env.py `_local_points` (554fb5e): a
    rotated object's box is then tight, a tumbled apple's is not the diagonal of its spec box."""
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    root = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise ValueError(f"No prim at {prim_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    ring = np.array([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)])
    pts = []
    for p in Usd.PrimRange(root):
        if not p.IsA(UsdGeom.Gprim) or not p.IsActive():
            continue
        if UsdGeom.Imageable(p).ComputePurpose() == UsdGeom.Tokens.guide and not (
                p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute("physics:collisionEnabled").Get() is not False):
            continue           # hidden helper geometry that does not collide
        if p.IsA(UsdGeom.Mesh):
            v = np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get() or [], float)
        elif p.IsA(UsdGeom.Cube):
            h = UsdGeom.Cube(p).GetSizeAttr().Get() / 2
            v = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)], float)
        elif p.IsA(UsdGeom.Cylinder) or p.IsA(UsdGeom.Capsule):
            g = UsdGeom.Cylinder(p) if p.IsA(UsdGeom.Cylinder) else UsdGeom.Capsule(p)
            r, hh = g.GetRadiusAttr().Get(), g.GetHeightAttr().Get() / 2
            axis = "XYZ".index(g.GetAxisAttr().Get() or "Z")
            v = np.concatenate([np.insert(ring * r, axis, z, axis=1) for z in (-hh, hh)])
        elif p.IsA(UsdGeom.Sphere):
            r = UsdGeom.Sphere(p).GetRadiusAttr().Get()
            v = np.concatenate([np.insert(ring * r, k, 0.0, axis=1) for k in range(3)])
        else:
            continue
        if not len(v):
            continue
        v = v[np.linspace(0, len(v) - 1, min(len(v), 400)).astype(int)]
        m, _ = cache.ComputeRelativeTransform(p, root)
        pts.append((np.c_[v, np.ones(len(v))] @ np.array(m))[:, :3])
    if not pts:
        raise ValueError(f"No geometry under {prim_path}")
    # the spawn scale sits on the root prim itself (not in the relative transforms)
    scale = np.linalg.norm(np.array(UsdGeom.Xformable(root).GetLocalTransformation())[:3, :3], axis=1)
    return np.concatenate(pts) * scale


def quat_matrix_wxyz(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def aabb(o, points, pos, quat):
    """Axis-aligned boxes (N, 2, 3) from the object's points and its poses (N, 3), (N, 4) wxyz."""
    out = np.zeros((len(pos), 2, 3))
    for i in range(len(pos)):
        if o["shape"] == "sphere":
            r = float(o["bounding_box"][0]) / 2
            out[i, 0], out[i, 1] = pos[i] - r, pos[i] + r
            continue
        world = points @ quat_matrix_wxyz(quat[i]).T + pos[i]
        out[i, 0], out[i, 1] = world.min(0), world.max(0)
    return out


def spawn_cfg(o):
    """The Isaac Lab spawn config of one spec object (isaaclab_biman_sim g1sim/assets.py spawn_cfg)."""
    import isaaclab.sim as sim_utils

    fixed = bool(o.get("fixed_base"))
    rigid = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=fixed, max_depenetration_velocity=1.0,
                                             solver_position_iteration_count=16, solver_velocity_iteration_count=1)
    material = sim_utils.RigidBodyMaterialCfg(static_friction=STATIC_FRICTION, dynamic_friction=DYNAMIC_FRICTION,
                                              restitution=0.0)
    rgba = o["asset"].get("rgba") or ((0.92, 0.92, 0.9, 1.0) if o["placeholder"] else (0.6, 0.6, 0.62, 1.0))
    bbox = [float(v) for v in o["bounding_box"]]
    if o["shape"] == "usd":  # chiral biman/assets.py spawn_cfg, registered rigid asset
        mass = sim_utils.MassPropertiesCfg(mass=o["mass"]) if o.get("mass") else \
            sim_utils.MassPropertiesCfg(density=o["density"])
        # asset.rgba on a registered asset: its look replaced by a plain colour (e.g. G1_DATA's toy eggs)
        look = (sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(o["asset"]["rgba"][:3]), roughness=0.6)
                if o["asset"].get("rgba") else None)
        return sim_utils.UsdFileCfg(usd_path=o["usd"], scale=tuple(o["scale"]), rigid_props=rigid, mass_props=mass,
                                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.004,
                                                                                     rest_offset=0.0),
                                    visual_material=look, activate_contact_sensors=True)
    if o["shape"] == "open_box":
        asset = o["asset"]
        if asset.get("storage_bin") is not None:
            # an open-fronted storage bin with a rim (asset key `storage_bin: {wall, rim_width, rim_height,
            # front_height, full_fraction}`, optional `emissive`; ours, see storage_bin.py): x along its depth, front at +x
            from .storage_bin import storage_bin_usd
            usd = storage_bin_usd(bbox, rgba, GENERATED_USD_DIR, emissive=asset.get("emissive", (0.0, 0.0, 0.0)),
                                  friction=(STATIC_FRICTION, DYNAMIC_FRICTION), **asset["storage_bin"])
        else:
            usd = open_box_usd(bbox, rgba, float(asset.get("wall", OPEN_BOX_WALL)), asset.get("front_height"))
        return sim_utils.UsdFileCfg(usd_path=usd, rigid_props=rigid,
                                    mass_props=sim_utils.MassPropertiesCfg(mass=float(asset.get("mass", OPEN_BOX_MASS))),
                                    activate_contact_sensors=True)
    common = dict(
        rigid_props=rigid, mass_props=sim_utils.MassPropertiesCfg(density=DENSITY),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0),
        physics_material=material,
        # asset.emissive (not in chiral's specs): a glow, for translucent or very bright real objects
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(rgba[:3]), roughness=0.5,
                                                    emissive_color=tuple(o["asset"].get("emissive", (0.0, 0.0, 0.0)))),
        activate_contact_sensors=True,
    )
    if o["shape"] == "cylinder" and o["asset"].get("dish"):
        # a real plate (asset key `dish: true`, not in chiral's specs): a shallow round dish of the box's size instead
        # of the flat USD cylinder, which also renders with about ten sides
        from ..sort_shapes.cube_spawner import DishCfg
        return DishCfg(diameter=min(bbox[:2]), height=bbox[2], **common)
    if o["shape"] == "cube" and o["asset"].get("oval_basket"):
        # the author's oval wicker basket with a handle (asset key `oval_basket: true`, `rim_height` [m], optional
        # `mass` [kg]; not in chiral's specs): the box is its length (x), width (y, under the handle) and total height
        from .oval_basket import oval_basket_usd
        return sim_utils.UsdFileCfg(usd_path=oval_basket_usd(bbox, o["asset"].get("rim_height", 0.11), rgba, GENERATED_USD_DIR),
                                    rigid_props=rigid, activate_contact_sensors=True,
                                    mass_props=sim_utils.MassPropertiesCfg(mass=float(o["asset"].get("mass", 0.15))))
    if o["shape"] == "cube" and o["asset"].get("lump"):
        # a soft wad (asset key `lump: true`, `mass` [kg], optional `seed`, `stiffness` [N/m]; not in chiral's specs):
        # an irregular rounded lump of the box's size with compliant contacts, so that the fingers sink into it a
        # little instead of meeting a hard body. A rigid stand-in for bubble wrap or a glove.
        from ..sort_shapes.cube_spawner import LumpCfg
        soft = sim_utils.RigidBodyMaterialCfg(
            static_friction=STATIC_FRICTION, dynamic_friction=DYNAMIC_FRICTION, restitution=0.0,
            compliant_contact_stiffness=float(o["asset"].get("stiffness", 2000.0)),
            compliant_contact_damping=float(o["asset"].get("damping", 20.0)))
        lump = dict(common, physics_material=soft, mass_props=sim_utils.MassPropertiesCfg(mass=float(o["asset"].get("mass", 0.01))))
        return LumpCfg(size=tuple(bbox), seed=int(o["asset"].get("seed", 0)), **lump)
    if o["shape"] == "cube" and o["asset"].get("toy_banana"):
        # the author's toy banana (asset key `toy_banana: true`, optional `mass` [kg]; not in chiral's specs): a round
        # curved tube of the box's size (x along it, y across the curve, z its thickness) instead of the box
        from ..sort_shapes.cube_spawner import ToyBananaCfg
        banana = dict(common, mass_props=sim_utils.MassPropertiesCfg(mass=float(o["asset"].get("mass", 0.04))))
        return ToyBananaCfg(length=bbox[0], width=bbox[1], thickness=bbox[2], **banana)
    if o["shape"] == "cylinder":
        return sim_utils.CylinderCfg(radius=min(bbox[:2]) / 2, height=bbox[2], axis="Z", **common)
    if o["shape"] == "sphere":
        return sim_utils.SphereCfg(radius=bbox[0] / 2, **common)
    return sim_utils.CuboidCfg(size=tuple(bbox), **common)


def open_box_usd(size, rgba, wall=OPEN_BOX_WALL, front_height=None):
    """An open-top box (floor and four walls, one rigid body with a compound collider of cubes) of outer size
    (x, y, z), centred on its box centre; written once per size and colour (isaaclab_biman_sim open_box_usd).
    `front_height` (ours): the wall at +x only this high, as the open front of a storage bin."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    sx, sy, sz = (round(float(v), 4) for v in size)
    col = tuple(round(float(c), 3) for c in rgba[:3])
    front = "" if front_height is None else f"_f{round(float(front_height), 4)}"
    path = GENERATED_USD_DIR / f"open_box_{sx}x{sy}x{sz}_w{wall}{front}_c{col[0]}-{col[1]}-{col[2]}.usda"
    if path.exists():
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/box")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim())
    look = UsdShade.Material.Define(stage, "/box/Looks/mat")
    shader = UsdShade.Shader.Define(stage, "/box/Looks/mat/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*col))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    look.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    physics = UsdShade.Material.Define(stage, "/box/Looks/physics")
    api = UsdPhysics.MaterialAPI.Apply(physics.GetPrim())
    api.CreateStaticFrictionAttr(STATIC_FRICTION)
    api.CreateDynamicFrictionAttr(DYNAMIC_FRICTION)
    api.CreateRestitutionAttr(0.0)
    parts = {"floor": ((0, 0, -sz / 2 + wall / 2), (sx, sy, wall)),
             "wall_xn": ((-sx / 2 + wall / 2, 0, 0), (wall, sy, sz)), "wall_xp": ((sx / 2 - wall / 2, 0, 0), (wall, sy, sz)),
             "wall_yn": ((0, -sy / 2 + wall / 2, 0), (sx, wall, sz)), "wall_yp": ((0, sy / 2 - wall / 2, 0), (sx, wall, sz))}
    if front_height is not None:
        fh = float(front_height)
        parts["wall_xp"] = ((sx / 2 - wall / 2, 0, -sz / 2 + fh / 2), (wall, sy, fh))
    for name, (centre, scale) in parts.items():
        cube = UsdGeom.Cube.Define(stage, f"/box/{name}")
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.XformCommonAPI(cube)
        xform.SetTranslate(Gf.Vec3d(*centre))
        xform.SetScale(Gf.Vec3f(*scale))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(look)
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(physics, UsdShade.Tokens.weakerThanDescendants, "physics")
    stage.GetRootLayer().Save()
    return str(path)
