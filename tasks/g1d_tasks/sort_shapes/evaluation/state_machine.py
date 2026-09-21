"""Pure tensor operations for ordered stage progression."""

import torch


def advance_ordered_stages(achieved, dwell_s, conditions, required_dwell_s, dt):
    """Advance monotonic stages while enforcing order and continuous dwell time.

    The tensors use ``(environment, object, stage)`` layout. A later stage may
    begin accumulating on the same control step that its prerequisite becomes
    achieved, but it can never be credited before that prerequisite.
    """
    if achieved.shape != dwell_s.shape or achieved.shape != conditions.shape:
        raise ValueError("achieved, dwell_s and conditions must have identical shapes")
    if required_dwell_s.shape != (achieved.shape[-1],):
        raise ValueError("required_dwell_s must contain one value per stage")
    if dt <= 0:
        raise ValueError("dt must be positive")

    next_achieved = achieved.clone()
    next_dwell = dwell_s.clone()
    newly_achieved = torch.zeros_like(achieved)
    for stage_index in range(achieved.shape[-1]):
        prerequisite = (
            torch.ones_like(conditions[:, :, stage_index])
            if stage_index == 0
            else next_achieved[:, :, stage_index - 1]
        )
        pending = ~next_achieved[:, :, stage_index]
        eligible = conditions[:, :, stage_index] & prerequisite & pending
        next_dwell[:, :, stage_index] = torch.where(
            pending,
            torch.where(
                eligible,
                next_dwell[:, :, stage_index] + dt,
                torch.zeros_like(next_dwell[:, :, stage_index]),
            ),
            next_dwell[:, :, stage_index],
        )
        gained = eligible & (
            next_dwell[:, :, stage_index] + 1e-7 >= required_dwell_s[stage_index]
        )
        next_achieved[:, :, stage_index] |= gained
        newly_achieved[:, :, stage_index] = gained
    return next_achieved, next_dwell, newly_achieved
