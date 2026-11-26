"""Mesh of the author's toy banana: a curved round tube that tapers to both ends.

The real toy (measured 2026-11-25 from the grasps on the real robot: the gripper pads stop 27 to 33 mm apart on it; its
length and curve from the head-camera images) is slimmer and less curved than the registered banana model, and a
per-axis scaling of that model flattens its cross-section. Local frame as the registered model, so that the same
poses apply: long axis x with the stem at +x, the curve in the xy plane bulging towards -y, centred on its bounding box.
"""
import numpy as np


def toy_banana(length, width, thickness, rings=28, sections=16, end_fraction=0.42):
    """Return (vertices, triangles, u): the tube of bounding box (length, width, thickness), `u` in [-1, 1] per vertex
    along the banana (-1 tip, +1 stem). The body is `thickness` across; the ends taper to `end_fraction` of it."""
    sag = max(width - thickness, 1e-4)                  # of the centreline, between its middle and its ends
    half = length / 2 - thickness / 4                   # half chord of the centreline
    for _ in range(8):                                  # the tapered ends shift the box a little: adjust until it fits
        vertices, faces, u_of = _tube(half, sag, thickness, rings, sections, end_fraction)
        extent = vertices.max(0) - vertices.min(0)
        half += (length - extent[0]) / 2; sag = max(sag + width - extent[1], 1e-4)
    vertices -= (vertices.min(0) + vertices.max(0)) / 2     # centred on the bounding box, like the primitives
    return vertices, faces, u_of


def _tube(half, sag, thickness, rings, sections, end_fraction):
    r0 = thickness / 2
    radius = lambda u: r0 * (1 - (1 - end_fraction) * np.abs(u) ** 3)
    big_r = (half ** 2 + sag ** 2) / (2 * sag)
    phi = np.arcsin(half / big_r)                       # half opening angle of the arc
    us = np.linspace(-1, 1, rings)
    ang = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    vertices, u_of = [], []
    for u in us:
        a = u * phi
        centre = np.array([big_r * np.sin(a), big_r * (1 - np.cos(a)) - sag, 0.0])     # middle at y = -sag
        normal = np.array([np.sin(a), -np.cos(a), 0.0])                                 # in-plane, away from the arc centre
        for t in ang:
            vertices.append(centre + radius(u) * (np.cos(t) * normal + np.sin(t) * np.array([0.0, 0.0, 1.0])))
            u_of.append(u)
    faces = []
    for i in range(rings - 1):
        for j in range(sections):
            a, b = i * sections + j, i * sections + (j + 1) % sections
            c, d = a + sections, b + sections
            faces += [[a, c, b], [b, c, d]]
    for ring, u, flip in ((0, -1.0, False), (rings - 1, 1.0, True)):               # rounded caps: one vertex past each end
        a = u * phi
        tangent = np.array([np.cos(a), np.sin(a), 0.0]) * np.sign(u)
        centre = np.array([big_r * np.sin(a), big_r * (1 - np.cos(a)) - sag, 0.0])
        vertices.append(centre + 0.5 * radius(u) * tangent); u_of.append(u); k = len(vertices) - 1
        for j in range(sections):
            p, q = ring * sections + j, ring * sections + (j + 1) % sections
            faces.append([k, q, p] if flip else [k, p, q])
    return np.array(vertices), np.array(faces), np.array(u_of)


def part(vertices, faces, u, low, high, grow=0.0):
    """The faces whose vertices all lie in low <= u <= high, as their own mesh, pushed `grow` m outwards from the
    banana's centreline plane point (for the coloured ends, drawn just outside the body)."""
    keep = np.all((u[faces] >= low) & (u[faces] <= high), axis=1)
    used = np.unique(faces[keep]); index = -np.ones(len(vertices), int); index[used] = np.arange(len(used))
    pts = vertices[used].copy()
    if grow:
        centre = pts.mean(0); pts = centre + (pts - centre) * (1 + grow / max(np.linalg.norm(pts - centre, axis=1).max(), 1e-6))
    return pts, index[faces[keep]]
