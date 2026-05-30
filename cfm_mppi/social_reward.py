# vendor/cfm_mppi/cfm_mppi/social_reward.py
"""Social CFM-bias reward functions.

Each fn is single-sample, pure-functional, vmap+grad compatible. The first
argument is `ego_controls: [ctrl_dim=2, horizon]` (the vendor contract — run_CFM
vmaps grad over the batch dim, so each call sees [2, H]). Every fn transposes to
[horizon, 2] internally as its first statement, then operates in [horizon, 2].
Returns scalar (negative cost = positive reward — CFM treats higher reward as better).

Sign discontinuities are sigmoid-gated (k≈10) per spec §3.1.
"""

from typing import NamedTuple

import torch

DT = 0.1
_SIGMOID_K = 10.0
_M_ALIGN = 0.3


def _controls_to_positions(ego_controls: torch.Tensor) -> torch.Tensor:
    """Cumulative integration: [horizon, 2] controls → [horizon, 2] xy positions.

    Robot frame, starting at origin. Receives the POST-transpose [horizon, 2] view.
    Same result as upstream reward.py cumsum(dim=1) on [ctrl_dim, horizon].
    """
    return torch.cumsum(ego_controls * DT, dim=0)


class NormSideStep(NamedTuple):
    """Per-(robot-batch, ped) decomposition of the norm passing-side step term.

    Every field is [B, n_peds] where B is the robot batch dim (CFM horizon H, MPPI
    n_samples, or 1 for the offline metric). Single source of truth shared by the CFM
    reward, the MPPI cost, and the side_compliance metric so the three cannot drift
    (spec v2 §3.1, §10). pen_term/coh_num/coh_den drive the cost; membership/dy drive
    the metric's qualify + compliance-sign.
    """

    pen_term: torch.Tensor  # membership * wrong-side penalty (the cost summand)
    membership: torch.Tensor  # moving * align * closing * inrange (metric qualify gate)
    coh_num: torch.Tensor  # moving * align * inrange (coherence numerator)
    coh_den: torch.Tensor  # moving * inrange         (coherence denominator)
    dy: torch.Tensor  # robot lateral offset in each ped's heading frame (metric sign)


def _norm_side_step(
    robot_xy: torch.Tensor,  # [B, 2] world-frame robot positions
    robot_vel: torch.Tensor,  # [B, 2] world-frame robot velocities
    ped_xy: torch.Tensor,  # [n_peds, 2] world-frame ped positions
    ped_vel: torch.Tensor,  # [n_peds, 2] world-frame ped velocities
    preferred_side: float = -1.0,
    v_min: float = 0.3,
    max_range: float = 5.0,
) -> "NormSideStep":
    """Crowd-robust norm passing-side membership + penalty, per (robot-batch, ped).

    All inputs are PHYSICAL world-frame tensors; each caller supplies robot_vel for its
    own dynamics (CFM: SI control == world velocity; MPPI: unicycle v*[cosθ,sinθ];
    metric: finite-diff of the dumped trajectory). The ONLY hard gate is `moving` on the
    PED speed — constant w.r.t. ego controls, so it is differentiable-safe (the reaction
    iso_below_speed / norm_yield hard-mask precedent). align/closing/inrange are smooth
    sigmoids. Spec v2 §3.1.
    """
    eps = 1e-8
    speed = torch.norm(ped_vel, dim=-1)  # [n_peds]
    moving = (speed > v_min).to(robot_xy.dtype).unsqueeze(0)  # [1, n_peds] HARD
    theta = torch.atan2(ped_vel[:, 1], ped_vel[:, 0])  # [n_peds] ped heading
    cos_t = torch.cos(-theta)
    sin_t = torch.sin(-theta)
    R = torch.stack(
        [torch.stack([cos_t, -sin_t], dim=-1), torch.stack([sin_t, cos_t], dim=-1)],
        dim=-2,
    )  # [n_peds, 2, 2]
    delta_world = robot_xy.unsqueeze(1) - ped_xy.unsqueeze(0)  # [B, n_peds, 2]
    delta = torch.einsum("pqr,bpr->bpq", R, delta_world)  # [B, n_peds, 2]
    dy = delta[..., 1]  # [B, n_peds] robot lateral in ped frame

    h_ped = ped_vel / (speed.unsqueeze(-1) + eps)  # [n_peds, 2]
    r_speed = torch.norm(robot_vel, dim=-1, keepdim=True)  # [B, 1]
    h_robot = robot_vel / (r_speed + eps)  # [B, 2]
    dot = torch.einsum("bd,pd->bp", h_robot, h_ped)  # [B, n_peds]
    align = torch.sigmoid(_SIGMOID_K * (-dot - _M_ALIGN))  # [B, n_peds] ONCOMING

    rel_v = robot_vel.unsqueeze(1) - ped_vel.unsqueeze(0)  # [B, n_peds, 2]
    ped_minus_robot = ped_xy.unsqueeze(0) - robot_xy.unsqueeze(1)  # [B, n_peds, 2]
    closing = torch.sigmoid(
        _SIGMOID_K * (ped_minus_robot * rel_v).sum(dim=-1)
    )  # [B, n_peds] APPROACHING (range-rate < 0)

    dist = torch.norm(delta_world, dim=-1)  # [B, n_peds]
    inrange = torch.sigmoid(_SIGMOID_K * (max_range - dist))  # [B, n_peds]

    pen = torch.nn.functional.softplus(
        _SIGMOID_K * (preferred_side * dy)
    )  # [B, n_peds] wrong-side penalty

    membership = moving * align * closing * inrange  # [B, n_peds]
    pen_term = membership * pen
    coh_num = moving * align * inrange
    coh_den = moving * inrange
    return NormSideStep(pen_term, membership, coh_num, coh_den, dy)


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
    v_min: float = 0.3,
    max_range: float = 5.0,
) -> torch.Tensor:
    """Crowd-robust norm passing-side CFM bias.

    Per horizon step: a GATED SUM over peds of (membership · wrong-side penalty), scaled by
    a corridor-coherence factor coh_t ∈ [0,1] (fraction of moving, in-range peds that are
    oncoming). C_norm = w_corridor · Σ_t coh_t · Σ_j w_j·pen_j; returns -C_norm (reward).
    Gates: moving (HARD, ped-speed) / oncoming (align margin) / closing / in-range — all
    via the shared _norm_side_step. Mirrors norm_side_cost term-for-term (locus parity).
    Spec v2 §3.1-3.3.
    """
    ego_controls = ego_controls.transpose(0, 1)  # [2,H] vendor contract -> [H,2]
    dtype = ego_controls.dtype
    ped_states = ped_states.to(dtype)
    ped_velocities = ped_velocities.to(dtype)
    xy = _controls_to_positions(ego_controls)  # [H, 2] robot positions
    robot_vel = ego_controls  # [H, 2] single-integrator SI == world velocity
    step = _norm_side_step(
        xy,
        robot_vel,
        ped_states,
        ped_velocities,
        preferred_side=preferred_side,
        v_min=v_min,
        max_range=max_range,
    )
    pen_sum = step.pen_term.sum(dim=-1)  # [H] gated sum over peds
    coh = step.coh_num.sum(dim=-1) / (step.coh_den.sum(dim=-1) + 1e-8)  # [H]
    g = coh * pen_sum  # [H]
    return -(w_corridor * g.sum())


def single_norm_yield_reward_fn(
    ego_controls: torch.Tensor,  # [ctrl_dim=2, horizon] SI velocities (vx, vy) — vendor contract
    ped_states: torch.Tensor,  # [n_peds, 2] current positions
    ped_velocities: torch.Tensor,  # [n_peds, 2]
    tau: float = 1.5,  # PET conflict time-window [s] (bell width)
    sigma: float = 0.5,  # "on the ped's path" spatial scale [m]
    v_min: float = 0.3,  # g_pedspeed threshold [m/s]
    max_range: float = 5.0,  # g_range cutoff [m] (matches proxemic)
    cross_min: float = 0.2,  # |sin(angle(robot_vel,ped_vel))| floor: suppress collinear
) -> torch.Tensor:
    """Temporal arrival-order (PET) yield reward at the robot-trajectory crossing.

    Resolves M1 (integration-risks.md): the prior CPA `robot_first` sigmoid was
    degenerate (t_robot ≡ t_ped ≡ ttca → constant 0.5). This scores the Post-
    Encroachment-Time at the point where the pedestrian's predicted constant-velocity
    path crosses the robot's ACTUAL rollout, found by projecting each rollout point onto
    the ped's CV line. PET = s* − t·DT (ped-arrival − robot-arrival).

    The who-first magnitude is the BELL softplus(PET)·exp(−PET²/2τ²): a hump peaked at a
    tight near-simultaneous crossing, decaying to ~0 BOTH for PET≫τ (robot cleared the
    crossing with room — no temporal conflict; whether it passed too close in front is the
    spatial PROXEMIC term's job, keeping the two terms orthogonal) AND for PET<0 (robot
    yielded behind). The non-monotonicity is intentional: it gives two low-cost basins
    (clear-ahead vs yield-behind) with the tight crossing as the costly ridge between, so
    the planner resolves a conflict by whichever it has room for. Locus-identical to
    norm_yield_cost. See spec docs/superpowers/specs/2026-05-30-norm-yield-temporal-design.md.

    R4: rolls out the WHOLE horizon and sums the per-step penalty so torch.func.grad
    flows to every control column. World/SI frame — CFM operates on SI velocities (no
    heading), and PET is frame-free. Returns unweighted −C_yield (weight applied
    externally via norm_yield_margin_coef).
    """
    ego_controls = ego_controls.transpose(
        0, 1
    )  # [2,H] vendor contract -> [H,2] internal
    H = ego_controls.shape[0]
    xr = _controls_to_positions(ego_controls)  # [H, 2] robot rollout
    t_arrival = (
        (torch.arange(H, dtype=ego_controls.dtype, device=ego_controls.device) + 1) * DT
    ).view(H, 1)  # [H,1] xr[i]=cumsum is reached AFTER i+1 steps → t=(i+1)*DT, not i*DT
    p0 = ped_states.unsqueeze(0)  # [1, n_peds, 2]
    vi = ped_velocities.unsqueeze(0)  # [1, n_peds, 2]
    r = xr.unsqueeze(1) - p0  # [H, n_peds, 2]
    vi_sq = (vi * vi).sum(dim=-1) + 1e-6  # [1, n_peds]
    s_star = (r * vi).sum(dim=-1) / vi_sq  # [H, n_peds] ped time nearest xr
    p_cross = p0 + vi * s_star.unsqueeze(-1)  # [H, n_peds, 2] nearest point on ped path
    miss = torch.norm(
        xr.unsqueeze(1) - p_cross, dim=-1
    )  # [H, n_peds] robot -> ped PATH
    pet = s_star - t_arrival  # [H, n_peds]
    dist = torch.norm(r, dim=-1)  # [H, n_peds] robot -> ped now
    pedspeed = torch.norm(ped_velocities, dim=-1).unsqueeze(0)  # [1, n_peds]
    # transversality: yield is a CROSSING term, so suppress collinear (head-on / following /
    # parallel) encounters — those belong to proxemic/danger. |sin(angle(robot_vel,ped_vel))|
    # via the 2D cross product; HARD gate so it survives run_CFM gradient normalization.
    ur = ego_controls / (torch.norm(ego_controls, dim=-1, keepdim=True) + 1e-6)  # [H,2]
    up = ped_velocities / (
        torch.norm(ped_velocities, dim=-1, keepdim=True) + 1e-6
    )  # [n_peds,2]
    sin_cross = (ur[:, 0:1] * up[:, 1] - ur[:, 1:2] * up[:, 0]).abs()  # [H, n_peds]
    penalty = (
        torch.nn.functional.softplus(pet)  # robot-first magnitude (PET>0 ⇒ cut-in)
        * torch.exp(
            -(pet * pet) / (2 * tau * tau)
        )  # tight-gap bell: ~0 for big lead OR yielded
        * torch.exp(-(miss * miss) / (2 * sigma * sigma))  # robot on the ped's path
        * torch.sigmoid(_SIGMOID_K * s_star)  # crossing is ahead of the ped (s* > 0)
        * torch.sigmoid(_SIGMOID_K * (max_range - dist))  # within range
        * torch.sigmoid(_SIGMOID_K * (pedspeed - v_min))  # ped is actually moving
        # HARD speed mask: the soft sigmoid above never reaches 0, and run_CFM
        # unit-normalizes this term's gradient (eval_utils.py), so a merely-attenuated
        # stopped ped renormalizes to full steering. A hard zero makes it truly inert
        # (proxemic owns the standing-ped case). Constant in ego_controls ⇒ no autograd
        # effect on the moving-ped gradient. Mirrored in norm_yield_cost (locus parity).
        * (pedspeed > v_min).to(pet.dtype)
        * (sin_cross > cross_min).to(
            pet.dtype
        )  # HARD transversality gate (crossing-only)
    )  # [H, n_peds]
    return -penalty.sum()  # R4: sum over horizon + peds; unweighted reward


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
