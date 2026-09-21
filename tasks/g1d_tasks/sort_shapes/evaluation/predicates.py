"""Pure geometric predicates shared by progress and stage evaluation."""

import torch


def _matrix_from_quat(quat):
    """Rotation matrices for scalar-first unit quaternions."""
    w, x, y, z = quat.unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ), dim=-1).reshape(*quat.shape[:-1], 3, 3)


def world_extent(quat, obj):
    """Axis-aligned world half-extent of one configured object."""
    rot = _matrix_from_quat(quat)
    if obj["kind"] == "cube":
        half = torch.as_tensor(obj["size"], device=quat.device, dtype=quat.dtype) / 2
        return (rot.abs() * half).sum(-1)
    axis = rot[:, :, 2]
    return obj["radius"] * (1 - axis.square()).clamp_min(0).sqrt() + obj["size"][2] / 2 * axis.abs()


def point_surface_distance(points_w, object_pos_w, object_quat_w, obj):
    """Distance from points to an oriented cuboid or finite cylinder surface.

    Points inside the object have distance zero. ``points_w`` has shape
    ``(num_envs, num_points, 3)``.
    """
    count = points_w.shape[1]
    relative = points_w - object_pos_w[:, None, :]
    quat = object_quat_w[:, None, :].expand(-1, count, -1)
    rotation = _matrix_from_quat(quat)
    local = torch.matmul(rotation.transpose(-1, -2), relative.unsqueeze(-1)).squeeze(-1)
    if obj["kind"] == "cube":
        half = torch.as_tensor(obj["size"], device=points_w.device, dtype=points_w.dtype) / 2
        return (local.abs() - half).clamp_min(0).norm(dim=-1)
    radial = local[..., :2].norm(dim=-1) - obj["radius"]
    axial = local[..., 2].abs() - obj["size"][2] / 2
    return torch.stack((radial, axial), dim=-1).clamp_min(0).norm(dim=-1)


def fully_inside_bin(local_pos, quat, obj, box, table_height, margin=0.0):
    """Whether the complete object is geometrically inside an open bin."""
    extent = world_extent(quat, obj)
    center = torch.as_tensor(box["center_xy"], device=local_pos.device, dtype=local_pos.dtype)
    half = torch.as_tensor(box["inner_size"][:2], device=local_pos.device, dtype=local_pos.dtype) / 2
    half = half - margin
    inside = ((local_pos[:, :2] - center).abs() + extent[:, :2] <= half).all(-1)
    bottom = table_height + box["wall_thickness"]
    inside &= local_pos[:, 2] - extent[:, 2] >= bottom - 0.004
    inside &= local_pos[:, 2] + extent[:, 2] <= bottom + box["inner_size"][2]
    return inside


def center_over_bin(local_pos, box, margin=0.0):
    """Whether an object's centre is horizontally over a bin opening."""
    center = torch.as_tensor(box["center_xy"], device=local_pos.device, dtype=local_pos.dtype)
    half = torch.as_tensor(box["inner_size"][:2], device=local_pos.device, dtype=local_pos.dtype) / 2
    return ((local_pos[:, :2] - center).abs() <= half - margin).all(-1)
