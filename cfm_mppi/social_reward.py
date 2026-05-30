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
    ped_states: torch.Tensor,  # [n_peds, 2] positions
    ped_velocities: torch.Tensor,  # [n_peds, 2]
    sigma_ref: float = 0.5,  # passing-side confidence scale [m]
    v_min: float = 0.3,  # min ped speed for a defined heading [m/s]
    max_range: float = 5.0,  # range gate [m] (matches proxemic)
) -> torch.Tensor:
    """Interaction-level legibility: reward early, decisive commitment to HOW the
    robot rounds each conflicting pedestrian.

    For each ped, sigma_t = signed offset of the robot rollout from the ped's
    constant-velocity path (sign = rounding side, |sigma| = miss distance = Social
    Momentum "confidence"). Saturated as clamp(sigma/sigma_ref, -1, 1) so legibility
    rewards a CLEAR side, not a wide detour. markup_t = 1.01^(T-t) front-loads early
    steps (Dragan). The net magnitude |sum_t markup_t * sigma_tilde| makes a dithering
    rollout self-cancel and is side-agnostic (norm_side owns WHICH side).

    RENORM SURVIVAL (codex H1/H2, mirrors the norm_yield lesson): run_CFM unit-
    normalizes this term's gradient (eval_utils.py:254), which ERASES any soft
    attenuation — only an EXACT-zero gradient stays inert after renormalization.
    So (a) saturation is a hard clamp (grad EXACTLY 0 once |sigma|>sigma_ref, no
    widening push for renorm to restore) not tanh (asymptotic, never zero); and
    (b) the range/speed gate is a HARD detached 0/1 mask (an out-of-range or slow
    ped contributes EXACTLY 0 grad, truly inert) not a sigmoid product (which never
    reaches 0 and renormalizes back to full steering).

    Grounding: Mavrogiannis Social Momentum (sign=side, magnitude=confidence) +
    Goyal 2026 (interaction-level = passing side) + Dragan 2013 (T-t front-loading).
    Markup is INTERNAL here (the net-magnitude form couples horizon steps, so the
    upstream external-markup multiply no longer applies). Returns the unweighted
    reward (higher = more legible); w_leg applied externally in run_CFM.
    """
    ego_controls = ego_controls.transpose(0, 1)  # [2,H] -> [H,2]
    H = ego_controls.shape[0]
    xr = _controls_to_positions(ego_controls)  # [H, 2] robot rollout
    t = (
        (
            (torch.arange(H, dtype=ego_controls.dtype, device=ego_controls.device) + 1)
            * DT  # xr[i]=cumsum is REACHED at (i+1)*DT — off-by-one norm_yield fixed (bd088f1)
        ).view(H, 1, 1)
    )  # [H,1,1] ped-propagation time, aligned to robot rollout step
    p0 = ped_states.unsqueeze(0)  # [1, n_peds, 2]
    vi = ped_velocities.unsqueeze(0)  # [1, n_peds, 2]
    speed = torch.norm(ped_velocities, dim=-1)  # [n_peds]
    h = ped_velocities / (speed.unsqueeze(-1) + 1e-8)  # [n_peds, 2] heading unit
    n = torch.stack([-h[:, 1], h[:, 0]], dim=-1)  # [n_peds, 2] left-normal to path
    p_t = p0 + vi * t  # [H, n_peds, 2] ped CV position at each rollout step
    r = xr.unsqueeze(1) - p_t  # [H, n_peds, 2] robot(step) - ped(step)
    sigma = (r * n.unsqueeze(0)).sum(dim=-1)  # [H, n_peds] signed offset from path
    sigma_tilde = torch.clamp(
        sigma / sigma_ref, -1.0, 1.0
    )  # hard saturation: grad EXACTLY 0 past sigma_ref (renorm-proof, not tanh)
    markup = (
        1.01
        ** torch.arange(H, dtype=ego_controls.dtype, device=ego_controls.device).flip(0)
    ).view(H, 1)  # [H,1] Dragan front-load, INTERNAL
    C = (markup * sigma_tilde).sum(dim=0)  # [n_peds] net early-weighted commitment
    dist = torch.norm(ped_states, dim=-1)  # [n_peds] robot(origin) -> ped now
    # HARD detached 0/1 gate (NOT a sigmoid product): out-of-range / slow ped -> EXACTLY
    # 0 grad, truly inert after run_CFM's per-term gradient normalization. Constant in
    # ego_controls (no grad through the comparison), same as norm_yield's masks.
    gate = (dist < max_range).to(sigma.dtype) * (speed > v_min).to(
        sigma.dtype
    )  # [n_peds] nearby AND moving
    return (gate * torch.sqrt(C * C + 1e-8)).sum()


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
