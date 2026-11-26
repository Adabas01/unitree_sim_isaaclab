"""An open-fronted storage bin with a rim (the green box of the conveyor task, a ProfiPlus Box 3), as one rigid body
with a compound collider of boxes.

Modelled from the author's photos of the real box and its datasheet sizes (2026-11-26), not from CAD data:
  - thin side and back walls, standing in from the outer size by the rim's overhang;
  - along the top of the side and back walls a rim that sticks out: on the real box a ledge with a raised lip, which
    is what the gripper closes on (in the conveyor recordings the pads stop 9.4 mm apart). Here the rim is a solid
    band of that width, so a gripper from above feels the same thickness; the lip above the ledge is not modelled;
  - the side walls come down towards the open front in two steps (the real ones in a curve) to a low front wall.
The real box also narrows towards its floor; this one has upright walls.
"""
from pathlib import Path


def storage_bin_parts(size, wall=0.003, rim_width=0.0094, rim_height=0.015, front_height=0.045, full_fraction=0.60):
    """{name: (centre, size)} of the boxes, in the frame of the outer box (x along the depth, open front at +x,
    z up, origin at the centre of the outer size)."""
    L, W, H = (float(v) for v in size)
    inset = rim_width - wall                      # the walls' outer face stands in by this much below the rim
    yw = W / 2 - inset - wall / 2                 # side wall centre
    xb = -L / 2 + inset + wall / 2                # back wall centre
    x_in0, x1 = -L / 2 + inset, -L / 2 + full_fraction * L
    x2 = x1 + (L / 2 - x1) / 2
    parts = {"floor": ((inset / 2, 0, -H / 2 + wall / 2), (L - inset, W - 2 * inset, wall)),
             "back": ((xb, 0, 0), (wall, W - 2 * inset, H)),
             "back_rim": ((-L / 2 + rim_width / 2, 0, H / 2 - rim_height / 2), (rim_width, W, rim_height)),
             "front": ((L / 2 - wall / 2, 0, -H / 2 + front_height / 2), (wall, W - 2 * inset, front_height))}
    steps = ((x_in0, x1, H), (x1, x2, front_height + 0.67 * (H - front_height)), (x2, L / 2, front_height + 0.30 * (H - front_height)))
    for side, sgn in (("l", 1.0), ("r", -1.0)):
        for k, (xa, xe, h) in enumerate(steps):
            parts[f"side_{side}{k}"] = (((xa + xe) / 2, sgn * yw, -H / 2 + h / 2), (xe - xa, wall, h))
        parts[f"rim_{side}"] = (((-L / 2 + x1) / 2, sgn * (W / 2 - rim_width / 2), H / 2 - rim_height / 2), (x1 + L / 2, rim_width, rim_height))
    return parts


def storage_bin_usd(size, rgba, out_dir, emissive=(0.0, 0.0, 0.0), friction=(1.0, 0.8), **shape):
    """Write the bin's USD once per size, shape and colour; returns its path. `shape`: storage_bin_parts arguments."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    col = tuple(round(float(c), 3) for c in rgba[:3])
    em = tuple(round(float(c), 3) for c in emissive)
    key = "_".join(f"{k}{round(float(v), 4)}" for k, v in sorted(shape.items()))
    name = "storage_bin_" + "x".join(str(round(float(v), 4)) for v in size) + f"_{key}_c{col[0]}-{col[1]}-{col[2]}_e{em[0]}-{em[1]}-{em[2]}.usda"
    path = Path(out_dir) / name
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
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*em))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    look.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    physics = UsdShade.Material.Define(stage, "/box/Looks/physics")
    api = UsdPhysics.MaterialAPI.Apply(physics.GetPrim())
    api.CreateStaticFrictionAttr(float(friction[0]))
    api.CreateDynamicFrictionAttr(float(friction[1]))
    api.CreateRestitutionAttr(0.0)
    for part, (centre, scale) in storage_bin_parts(size, **shape).items():
        cube = UsdGeom.Cube.Define(stage, f"/box/{part}")
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.XformCommonAPI(cube)
        xform.SetTranslate(Gf.Vec3d(*(float(v) for v in centre)))
        xform.SetScale(Gf.Vec3f(*(float(v) for v in scale)))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(look)
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(physics, UsdShade.Tokens.weakerThanDescendants, "physics")
    stage.GetRootLayer().Save()
    return str(path)
