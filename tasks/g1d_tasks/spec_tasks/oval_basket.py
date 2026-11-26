"""An oval basket with a handle, as a generated USD (asset key `oval_basket: true`; not in chiral's specs).

Made for the author's wicker basket (his photo and measures of 2026-11-26): an oval rim of 24 x 17 cm, 11 cm high, and
a handle over the middle of the long sides, 25 cm high in all. Frame: x along the long side, y along the short side
(the handle arches in the y-z plane), z up, origin at the centre of the bounding box (length, width, total height).
One rigid body; every wall segment, the floor and every handle segment is its own convex collider, so that objects
fall into the basket and the handle is in the way of the hands as on the real one. The base is smaller than the rim
(`base_fraction`, read off the photo, not measured).
"""
import numpy as np

SIDES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))     # faces of a hexahedron


def oval_basket_parts(length, width, height, rim_height, base_fraction=0.72, wall=0.006, floor=0.006, handle_width=0.02,
                      handle_thickness=0.01, segments=28, handle_segments=12):
    """[(name, vertices (n, 3), faces)] of the basket's convex pieces."""
    a, b, z0 = length / 2, width / 2, -height / 2
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)

    def ring(sa, sb, z):
        return np.c_[sa * np.cos(angles), sb * np.sin(angles), np.full(segments, z)]

    ob, ot = ring(a * base_fraction, b * base_fraction, z0), ring(a, b, z0 + rim_height)
    ib, it = ring(a * base_fraction - wall, b * base_fraction - wall, z0), ring(a - wall, b - wall, z0 + rim_height)
    parts = []
    for i in range(segments):
        j = (i + 1) % segments
        parts.append((f"wall_{i:02d}", np.array([ob[i], ob[j], ib[j], ib[i], ot[i], ot[j], it[j], it[i]]), SIDES))
    # floor: an oval plate inside the wall, as wide as the wall's inner side at its top face
    grow = floor / rim_height * (1 - base_fraction)
    low, high = ring(a * base_fraction - wall, b * base_fraction - wall, z0), ring(a * (base_fraction + grow) - wall, b * (base_fraction + grow) - wall, z0 + floor)
    faces = [tuple(range(segments - 1, -1, -1)), tuple(range(segments, 2 * segments))]
    faces += [(i, (i + 1) % segments, segments + (i + 1) % segments, segments + i) for i in range(segments)]
    parts.append(("floor", np.r_[low, high], faces))
    # handle: half an ellipse from rim to rim over the short side, apex at the top of the bounding box
    phi = np.linspace(0, np.pi, handle_segments + 1); rise = height - rim_height - handle_thickness / 2
    centre = np.c_[np.zeros_like(phi), (b - wall / 2) * np.cos(phi), z0 + rim_height + rise * np.sin(phi)]
    tangent = np.c_[np.zeros_like(phi), -(b - wall / 2) * np.sin(phi), rise * np.cos(phi)]
    normal = np.c_[np.zeros_like(phi), tangent[:, 2], -tangent[:, 1]]; normal /= np.linalg.norm(normal, axis=1)[:, None]

    def section(k):
        c, n, w = centre[k], normal[k] * handle_thickness / 2, np.array([handle_width / 2, 0, 0])
        return [c - w - n, c + w - n, c + w + n, c - w + n]
    for k in range(handle_segments):
        parts.append((f"handle_{k:02d}", np.array(section(k) + section(k + 1)), SIDES))
    return parts


def oval_basket_usd(size, rim_height, rgba, directory, friction=(1.0, 0.8)):
    """Write the basket once per size and colour; returns the USD path."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    sx, sy, sz = (round(float(v), 4) for v in size); col = tuple(round(float(c), 3) for c in rgba[:3])
    path = directory / f"oval_basket_{sx}x{sy}x{sz}_rim{round(float(rim_height), 4)}_c{col[0]}-{col[1]}-{col[2]}.usda"
    if path.exists():
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/basket"); stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()); UsdPhysics.MassAPI.Apply(root.GetPrim())
    look = UsdShade.Material.Define(stage, "/basket/Looks/mat"); shader = UsdShade.Shader.Define(stage, "/basket/Looks/mat/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*col))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    look.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    physics = UsdShade.Material.Define(stage, "/basket/Looks/physics"); api = UsdPhysics.MaterialAPI.Apply(physics.GetPrim())
    api.CreateStaticFrictionAttr(friction[0]); api.CreateDynamicFrictionAttr(friction[1]); api.CreateRestitutionAttr(0.0)
    for name, vertices, faces in oval_basket_parts(sx, sy, sz, float(rim_height)):
        mesh = UsdGeom.Mesh.Define(stage, f"/basket/{name}")
        mesh.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in vertices])
        mesh.CreateFaceVertexCountsAttr([len(f) for f in faces]); mesh.CreateFaceVertexIndicesAttr([int(i) for f in faces for i in f])
        mesh.CreateDoubleSidedAttr(True); mesh.CreateSubdivisionSchemeAttr("none")
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()); UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(look)
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(physics, UsdShade.Tokens.weakerThanDescendants, "physics")
    stage.GetRootLayer().Save()
    return str(path)
