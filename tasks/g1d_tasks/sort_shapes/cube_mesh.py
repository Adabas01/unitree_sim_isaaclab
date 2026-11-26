"""Mesh of the real sorting cubes: chamfered edges and corners, and a round hole in each face.

The real cubes (measured by the author, 2026-11-24) are 48 mm across, each face has a flat square of 33 mm, and a
hole of 11 mm diameter sits in the middle of every face. The outer shape is the convex hull of the six flat squares
(square faces, rectangular edge chamfers, triangular corners). The holes are modelled as recesses of `hole_depth`
for the cameras; collision uses the convex hull, so they do not change the contact.
"""
import itertools

import numpy as np


def chamfered_cube(size, face_flat, hole_diameter=0.0, hole_depth=0.012, segments=24):
    """Return (vertices, triangles) with one vertex triple per triangle (flat shading) and outward normals.

    `segments` must be a multiple of 8, so that the ring of each face hits the corners of its flat square."""
    h, f, r = size / 2, face_flat / 2, hole_diameter / 2
    tris = []           # (triangle 3x3, point the normal must point away from)
    centre = np.zeros(3)

    def frame(axis, sign):
        n = np.zeros(3); n[axis] = sign
        u = np.zeros(3); u[(axis + 1) % 3] = 1.0
        v = np.zeros(3); v[(axis + 2) % 3] = 1.0
        return n, u, v

    for axis, sign in itertools.product(range(3), (-1.0, 1.0)):
        n, u, v = frame(axis, sign)
        c = n * h
        if r <= 0:
            q = [c + f * (a * u + b * v) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            tris += [((q[0], q[1], q[2]), centre), ((q[0], q[2], q[3]), centre)]
            continue
        ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        ring = [c + r * (np.cos(a) * u + np.sin(a) * v) for a in ang]
        # the same directions pushed out to the border of the flat square
        border = [c + f * (np.cos(a) * u + np.sin(a) * v) / max(abs(np.cos(a)), abs(np.sin(a))) for a in ang]
        bottom = c - n * hole_depth
        for i in range(segments):
            j = (i + 1) % segments
            tris += [((ring[i], border[i], border[j]), centre), ((ring[i], border[j], ring[j]), centre)]
            # wall of the hole: normal towards the hole's axis; floor of the hole: normal out of the face
            low_i, low_j = ring[i] - n * hole_depth, ring[j] - n * hole_depth
            away = 2 * (ring[i] + ring[j]) / 2 - (c - n * hole_depth / 2)     # a point outside the wall
            tris += [((ring[i], ring[j], low_j), away), ((ring[i], low_j, low_i), away)]
            tris.append(((bottom, low_i, low_j), bottom - n))
    # edge chamfers: between the flat squares of two neighbouring faces
    for a1, a2 in itertools.combinations(range(3), 2):
        a3 = 3 - a1 - a2
        for s1, s2 in itertools.product((-1.0, 1.0), repeat=2):
            p = []
            for s3 in (-1.0, 1.0):
                x = np.zeros(3); x[a1], x[a2], x[a3] = s1 * h, s2 * f, s3 * f; p.append(x)
            for s3 in (1.0, -1.0):
                x = np.zeros(3); x[a1], x[a2], x[a3] = s1 * f, s2 * h, s3 * f; p.append(x)
            tris += [((p[0], p[1], p[2]), centre), ((p[0], p[2], p[3]), centre)]
    # corner triangles
    for s in itertools.product((-1.0, 1.0), repeat=3):
        s = np.array(s)
        p = [s * np.where(np.arange(3) == k, h, f) for k in range(3)]
        tris.append(((p[0], p[1], p[2]), centre))

    vertices, faces = [], []
    for (a, b, c3), inside in tris:
        if np.dot(np.cross(b - a, c3 - a), (a + b + c3) / 3 - inside) < 0:
            b, c3 = c3, b
        faces.append([len(vertices), len(vertices) + 1, len(vertices) + 2]); vertices += [a, b, c3]
    return np.array(vertices), np.array(faces)
