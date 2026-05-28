# vendor/cfm_mppi/cfm_mppi/social_reward.py
"""Social CFM-bias reward functions.

Each fn is single-sample, pure-functional, vmap+grad compatible. The first
argument is `ego_controls: [ctrl_dim=2, horizon]` (the vendor contract — run_CFM
vmaps grad over the batch dim, so each call sees [2, H]). Every fn transposes to
[horizon, 2] internally as its first statement, then operates in [horizon, 2].
Returns scalar (negative cost = positive reward — CFM treats higher reward as better).

Sign discontinuities are sigmoid-gated (k≈10) per spec §3.1.
"""

import torch

DT = 0.1
_SIGMOID_K = 10.0


def _controls_to_positions(ego_controls: torch.Tensor) -> torch.Tensor:
    """Cumulative integration: [horizon, 2] controls → [horizon, 2] xy positions.

    Robot frame, starting at origin. Receives the POST-transpose [horizon, 2] view.
    Same result as upstream reward.py cumsum(dim=1) on [ctrl_dim, horizon].
    """
    return torch.cumsum(ego_controls * DT, dim=0)


def single_proxemic_reward_fn(
    ego_controls: torch.Tensor,  # [ctrl_dim=2, horizon] controls (vx, vy) — vendor contract
    ped_states: torch.Tensor,  # [n_peds, 2] positions
    ped_velocities: torch.Tensor,  # [n_peds, 2]
    sigma_front_base: float = 1.0,
    sigma_front_speed: float = 0.45,
    sigma_rear: float = 0.25,
    sigma_side: float = 0.5,
    max_range: float = 5.0,
) -> torch.Tensor:
    """Kirby asymmetric Gaussian, Neggers speed-adaptive sigma_front."""
    ego_controls = ego_controls.transpose(
        0, 1
    )  # [2,H] vendor contract -> [H,2] internal
    xy = _controls_to_positions(ego_controls)  # [H, 2]
    delta_world = xy.unsqueeze(1) - ped_states.unsqueeze(0)  # [H, n_peds, 2]
    speed = torch.norm(ped_velocities, dim=-1, keepdim=True)  # [n_peds, 1]
    theta = torch.atan2(ped_velocities[:, 1], ped_velocities[:, 0])  # [n_peds]
    cos_t = torch.cos(-theta)
    sin_t = torch.sin(-theta)
    R = torch.stack(
        [torch.stack([cos_t, -sin_t], dim=-1), torch.stack([sin_t, cos_t], dim=-1)],
        dim=-2,
    )  # [n_peds, 2, 2]
    delta = torch.einsum("npq,hnq->hnp", R, delta_world)  # [H, n_peds, 2]
    dx, dy = delta[..., 0], delta[..., 1]
    sigma_front = sigma_front_base + sigma_front_speed * speed.squeeze(-1)  # [n_peds]
    front_mask = torch.sigmoid(_SIGMOID_K * dx)  # [H, n_peds]
    sigma_along = front_mask * sigma_front + (1 - front_mask) * sigma_rear
    cost = torch.exp(
        -(dx * dx) / (2 * sigma_along * sigma_along + 1e-8)
        - (dy * dy) / (2 * sigma_side * sigma_side)
    )
    dist_world = torch.norm(delta_world, dim=-1)  # [H, n_peds]
    range_mask = torch.sigmoid(_SIGMOID_K * (max_range - dist_world))
    return -(cost * range_mask).sum()


def single_legibility_reward_fn(
    ego_controls: torch.Tensor,  # [ctrl_dim=2, horizon] — vendor contract
    ped_states: torch.Tensor,  # [n_peds, 2] — unused, signature-matched for vmap
    ped_velocities: torch.Tensor,  # [n_peds, 2] — unused
    goal_dir: torch.Tensor,  # [2] vector toward goal in robot frame
) -> torch.Tensor:
    """Reward alignment of ego velocity with goal direction.

    Markup-weighting is applied EXTERNALLY in run_CFM (spec §3.2). Returns
    unweighted scalar reward.
    """
    ego_controls = ego_controls.transpose(
        0, 1
    )  # [2,H] vendor contract -> [H,2] internal
    goal_unit = goal_dir / (torch.norm(goal_dir) + 1e-8)
    speed = torch.norm(ego_controls, dim=-1, keepdim=True)  # [H, 1]
    ego_unit = ego_controls / (speed + 1e-8)
    alignment = (ego_unit * goal_unit).sum(dim=-1)  # [H] cosine similarity
    # touch unused ped args so vmap doesn't choke on closed-over tensors
    _ = ped_states.sum() * 0.0 + ped_velocities.sum() * 0.0
    return alignment.sum() + _


def single_norm_side_reward_fn(
    ego_controls: torch.Tensor,  # [ctrl_dim=2, horizon] — vendor contract
    ped_states: torch.Tensor,  # [n_peds, 2]
    ped_velocities: torch.Tensor,  # [n_peds, 2]
    preferred_side: float = -1.0,  # -1 = right-pass (US/EU)
    w_corridor: float = 1.0,
) -> torch.Tensor:
    """Kalenberg asymmetric softplus on lateral offset vs preferred passing side."""
    ego_controls = ego_controls.transpose(
        0, 1
    )  # [2,H] vendor contract -> [H,2] internal
    xy = _controls_to_positions(ego_controls)  # [H, 2]
    # transform to each ped's frame
    theta = torch.atan2(ped_velocities[:, 1], ped_velocities[:, 0])
    cos_t = torch.cos(-theta)
    sin_t = torch.sin(-theta)
    R = torch.stack(
        [torch.stack([cos_t, -sin_t], dim=-1), torch.stack([sin_t, cos_t], dim=-1)],
        dim=-2,
    )
    delta_world = xy.unsqueeze(1) - ped_states.unsqueeze(0)
    delta = torch.einsum("npq,hnq->hnp", R, delta_world)
    dy = delta[..., 1]
    # asymmetric softplus: penalize wrong-side
    wrong_side = torch.nn.functional.softplus(_SIGMOID_K * (preferred_side * dy))
    # consider only peds in front (dx > 0)
    dx = delta[..., 0]
    front_mask = torch.sigmoid(_SIGMOID_K * dx)
    return -(wrong_side * front_mask * w_corridor).sum()


def single_norm_yield_reward_fn(
    ego_controls: torch.Tensor,  # [ctrl_dim=2, horizon] SI velocities (vx, vy) — vendor contract
    ped_states: torch.Tensor,  # [n_peds, 2]
    ped_velocities: torch.Tensor,  # [n_peds, 2]
    T_safe: float = 1.5,
    R_conflict: float = 1.0,
) -> torch.Tensor:
    """TTCP-asymmetric yield reward. Penalize robot crossing a conflict point first.

    R4 (horizon aggregation): rolls out ego positions across the WHOLE horizon and
    sums the per-step penalty, so torch.func.grad flows to every control column.
    R1 (per-agent CP): closest points computed per agent at t*; miss-distance gates a
    genuine near-conflict. No heading rotation — CFM operates on SI velocities.

    >>> MATH REVIEW REQUIRED (M1 / integration-risks.md) <<<
    The `robot_first` term below uses CPA-based per-agent times. But at the closest
    point of approach BOTH agents reach their CP simultaneously, so t_robot == t_ped ==
    ttca by construction and `sigmoid(K*(t_ped - t_robot)) == 0.5` — the passing-order
    asymmetry DEGENERATES to a constant. A correct "who yields" cost needs a SHARED
    spatial conflict point (path-ray intersection), then arrival times to THAT point
    differ. Leaving the CPA form as a placeholder; `conflict_gate` + `ttca_gate` are
    valid, but `robot_first` is NOT meaningful until the conflict-point definition is
    replaced. test_norm_yield_penalizes_robot_first xfails against this placeholder.
    """
    ego_controls = ego_controls.transpose(0, 1)  # [2,H] -> [H,2] internal
    H = ego_controls.shape[0]
    ego_pos = _controls_to_positions(ego_controls)  # [H, 2]
    t_idx = torch.arange(H, dtype=ego_controls.dtype).view(H, 1, 1)  # [H,1,1]
    ped_pos_t = ped_states.unsqueeze(0) + ped_velocities.unsqueeze(0) * (
        t_idx * DT
    )  # [H, n_peds, 2]
    r = ego_pos.unsqueeze(1) - ped_pos_t  # [H, n_peds, 2] robot-ped
    v_rel = ego_controls.unsqueeze(1) - ped_velocities.unsqueeze(0)  # [H, n_peds, 2]
    v_rel_sq = (v_rel * v_rel).sum(dim=-1) + 1e-6  # [H, n_peds]
    ttca = -(r * v_rel).sum(dim=-1) / v_rel_sq  # [H, n_peds]
    # per-agent closest points at t* (R1)
    robot_cp = ego_pos.unsqueeze(1) + ego_controls.unsqueeze(1) * ttca.unsqueeze(
        -1
    )  # [H,n_peds,2]
    ped_cp = ped_pos_t + ped_velocities.unsqueeze(0) * ttca.unsqueeze(
        -1
    )  # [H,n_peds,2]
    miss = torch.norm(robot_cp - ped_cp, dim=-1)  # [H, n_peds]
    speed_robot = torch.norm(ego_controls, dim=-1, keepdim=True) + 1e-6  # [H, 1]
    speed_ped = torch.norm(ped_velocities, dim=-1).unsqueeze(0) + 1e-6  # [1, n_peds]
    t_robot = (
        torch.norm(robot_cp - ego_pos.unsqueeze(1), dim=-1) / speed_robot
    )  # [H,n_peds] (== ttca; see FLAG)
    t_ped = (
        torch.norm(ped_cp - ped_pos_t, dim=-1) / speed_ped
    )  # [H,n_peds] (== ttca; see FLAG)
    robot_first = torch.sigmoid(
        _SIGMOID_K * (t_ped - t_robot)
    )  # DEGENERATE placeholder — see FLAG
    conflict_gate = torch.sigmoid(
        _SIGMOID_K * (R_conflict - miss)
    )  # genuine near-miss (valid)
    ttca_gate = torch.sigmoid(_SIGMOID_K * (T_safe - ttca))  # imminent only (valid)
    per_step = (robot_first * conflict_gate * ttca_gate).sum(dim=-1)  # [H]
    return -per_step.sum()  # R4: sum over horizon


def single_group_reward_fn(
    ego_controls: torch.Tensor,  # [ctrl_dim=2, horizon] — vendor contract
    ped_states: torch.Tensor,  # [n_peds, 2]
    group_pairs: torch.Tensor,  # [n_groups, 2] int indices into ped_states
    sigma_group: float = 0.3,
) -> torch.Tensor:
    """Distance-to-group-line Gaussian penalty (penalize crossing between paired peds)."""
    ego_controls = ego_controls.transpose(
        0, 1
    )  # [2,H] vendor contract -> [H,2] internal
    xy = _controls_to_positions(ego_controls)  # [H, 2]
    a = ped_states[group_pairs[:, 0]]  # [n_groups, 2]
    b = ped_states[group_pairs[:, 1]]  # [n_groups, 2]
    ab = b - a  # [n_groups, 2]
    ab_len_sq = (ab * ab).sum(dim=-1) + 1e-6  # [n_groups]
    ax = xy.unsqueeze(1) - a.unsqueeze(0)  # [H, n_groups, 2]
    t = (ax * ab.unsqueeze(0)).sum(dim=-1) / ab_len_sq  # [H, n_groups]
    t_clamped = torch.clamp(t, 0.0, 1.0)
    proj = a.unsqueeze(0) + t_clamped.unsqueeze(-1) * ab.unsqueeze(0)
    perp = xy.unsqueeze(1) - proj
    d = torch.norm(perp, dim=-1)  # [H, n_groups]
    cost = torch.exp(-(d * d) / (2 * sigma_group * sigma_group))
    return -cost.sum()
