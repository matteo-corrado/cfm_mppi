import torch
from cfm_mppi.reward import single_cbf_reward_fn_pairwise, single_goal_reward_fn
from cfm_mppi.social_reward import (
    single_proxemic_reward_fn,
    single_legibility_reward_fn,
    single_norm_side_reward_fn,
    single_norm_yield_reward_fn,
    single_group_reward_fn,
)
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CFMConfig:
    ode_times: List[float]
    dt: float = 0.1
    agent_radius: float = 0.5
    space_scale: float = 10.0
    safe_margin_coefs: Optional[List[float]] = None
    goal_margin_coef: float = 0.1
    device: str = "cuda"
    # NEW — social CFM-bias coefs (0.0 = term disabled; matches safe_margin_coefs pattern)
    proxemic_margin_coef: float = 0.0
    legibility_margin_coef: float = 0.0
    norm_side_margin_coef: float = 0.0
    norm_yield_margin_coef: float = 0.0
    group_margin_coef: float = 0.0
    # NEW — per-term scalar params (closed over at grad-fn registration, R3 — the uniform
    # 3-arg vmap call shape leaves no positional slot for them)
    norm_side_preferred_side: float = -1.0  # -1 = right-pass (US/EU)
    norm_yield_tau: float = 1.5  # PET conflict time-window [s]
    norm_yield_sigma: float = 0.5  # PET "on the ped's path" spatial scale [m]
    # NEW — group topology: [n_groups, 2] int indices into peds; None/empty = term off
    group_pairs: Optional[torch.Tensor] = None


def _social_now_index(seq_len, control_history):
    """Index of the 'current' pedestrian frame in a [.., horizon] sequence laid
    out as [committed-history | future] (synthesize_control prepends
    obstacle_state_history when control history exists). 'Now' is the first future
    frame = the committed-history length (0 when there is no history). Clamped to the
    sequence. Replaces the original obs_positions[..., j] (ODE-step index, wrong axis)
    and the later [..., 0] (the oldest committed frame, stale for t>=1)."""
    idx = control_history.shape[-1] if control_history is not None else 0
    return min(idx, seq_len - 1)


def _social_ped_snapshot(obs_positions, obs_velocities, control_history, dt):
    """Current pedestrian (position, velocity) for the social reward, recovered from
    the [committed-history | future] sequence run_CFM receives.

    synthesize_control builds the future half as ``current + cumsum(vel * dt)``, so the
    first future column (index = committed-history length, or 0 with no history) holds
    ``current + vel*dt`` -- one step AHEAD, not the current frame. Remove one velocity
    step to recover the exact current frame. Position and velocity are scaled
    identically (``/ space_scale`` in run_CFM), velocity is translation-invariant so the
    ego-relative shift cancels, and the future half repeats the current velocity, so
    ``obs_velocities[..., i]`` IS the current velocity and the subtraction is exact. The
    CBF path consumes the full time-aligned sequence separately and is unaffected.

    Coupled to synthesize_control's cumsum construction; pinned by
    tests/social/test_cfm_social_snapshot.py.
    """
    i = _social_now_index(obs_positions.shape[-1], control_history)
    vel = obs_velocities[..., i]
    pos = obs_positions[..., i] - vel * dt
    return pos, vel


def run_CFM(
    model,
    config: CFMConfig,
    noisy_action_seq,
    noise_level,
    start_pos,
    goal_pos,
    obs_positions,
    obs_velocities,
    control_history=None,
    sink=None,
):
    start_pos = start_pos / config.space_scale
    goal_pos = goal_pos / config.space_scale
    obs_positions = obs_positions / config.space_scale
    obs_velocities = obs_velocities / config.space_scale
    rad = config.agent_radius / config.space_scale

    if control_history is not None:
        control_history = control_history / config.space_scale

    start_batch = start_pos.repeat(noisy_action_seq.shape[0], 1)
    goal_batch = goal_pos.repeat(noisy_action_seq.shape[0], 1)
    obs_positions = obs_positions.squeeze(0)
    obs_velocities = obs_velocities.squeeze(0)
    batch_size = noisy_action_seq.shape[0]

    if config.safe_margin_coefs is not None:
        num_coef = len(config.safe_margin_coefs)
        size_coef = batch_size // num_coef
        safe_coef = torch.zeros(batch_size, 1, 1, device=noisy_action_seq.device)
        for i in range(num_coef):
            safe_coef[size_coef * i : size_coef * (i + 1), 0, 0] = (
                config.safe_margin_coefs[i]
            )

    single_cbf_grad_fn = torch.func.grad(single_cbf_reward_fn_pairwise)
    single_goal_grad_fn = torch.func.grad(single_goal_reward_fn)

    batched_cbf_grad_fn = torch.vmap(single_cbf_grad_fn, in_dims=(0, None, None, None))
    batched_goal_grad_fn = torch.vmap(single_goal_grad_fn, in_dims=(0, None))

    # NEW — social grad fns (one entry per enabled social term; skip compile if coef = 0)
    social_grad_fns = {}
    if config.proxemic_margin_coef > 0:
        social_grad_fns["proxemic"] = torch.vmap(
            torch.func.grad(single_proxemic_reward_fn),
            in_dims=(0, None, None),
        )
    if config.legibility_margin_coef > 0:
        social_grad_fns["legibility"] = torch.vmap(
            torch.func.grad(single_legibility_reward_fn),
            in_dims=(0, None, None, None),
        )
    if config.norm_side_margin_coef > 0:

        def _ns(
            ego_controls,
            ped_states,
            ped_velocities,
            _side=config.norm_side_preferred_side,
        ):
            return single_norm_side_reward_fn(
                ego_controls, ped_states, ped_velocities, preferred_side=_side
            )

        social_grad_fns["norm_side"] = torch.vmap(
            torch.func.grad(_ns),
            in_dims=(0, None, None),
        )
    if config.norm_yield_margin_coef > 0:

        def _ny(
            ego_controls,
            ped_states,
            ped_velocities,
            _tau=config.norm_yield_tau,
            _sigma=config.norm_yield_sigma,
        ):
            return single_norm_yield_reward_fn(
                ego_controls, ped_states, ped_velocities, tau=_tau, sigma=_sigma
            )

        social_grad_fns["norm_yield"] = torch.vmap(
            torch.func.grad(_ny),
            in_dims=(0, None, None),
        )
    if config.group_margin_coef > 0:
        social_grad_fns["group"] = torch.vmap(
            torch.func.grad(single_group_reward_fn),
            in_dims=(0, None, None),
        )

    for j in range(len(config.ode_times)):
        if control_history is not None:
            noisy_action_seq[:, :, : control_history.shape[-1]] = control_history
        t_next = torch.tensor([config.ode_times[j]], device=config.device)

        u_t_pred = model(
            noisy_action_seq, noise_level, start=start_batch, goal=goal_batch
        )
        x_1_pred = noisy_action_seq + (1 - noise_level) * u_t_pred

        if control_history is not None:
            x_1_pred[:, :, : control_history.shape[-1]] = control_history

        grad_cbf = batched_cbf_grad_fn(x_1_pred, obs_positions, obs_velocities, rad)
        grad_goal = batched_goal_grad_fn(x_1_pred, goal_pos.squeeze(0))

        if control_history is not None:
            mask = torch.ones_like(grad_cbf, device=config.device)
            mask[:, :, : control_history.shape[-1]] = 0
            grad_cbf = grad_cbf * mask
            grad_goal = grad_goal * mask

        u_norm = torch.norm(u_t_pred, keepdim=True)
        grad_cbf_norm = torch.norm(grad_cbf, keepdim=True)
        grad_goal_norm = torch.norm(grad_goal, keepdim=True)

        normalized_grad_cbf = grad_cbf * u_norm / (grad_cbf_norm + 1e-8)
        normalized_grad_goal = grad_goal * u_norm / (grad_goal_norm + 1e-8)

        markup = 1.01 ** torch.arange(
            0, noisy_action_seq.shape[-1], device=config.device
        ).flip(0).unsqueeze(0).unsqueeze(0)

        # NEW — social terms
        social_terms = []
        SOCIAL_TERM_META = [
            ("proxemic", False),
            ("legibility", True),
            ("norm_side", False),
            ("norm_yield", False),
            ("group", False),
        ]
        for name, apply_markup in SOCIAL_TERM_META:
            if name not in social_grad_fns:
                continue
            # Collapse the [n_peds, 2, horizon] sequence to the CURRENT [n_peds, 2]
            # snapshot the social reward fns expect. synthesize_control lays it out
            # as [committed-history | future] and builds the future half via
            # current + cumsum(vel*dt), so the first future column (index = history
            # length, 0 when none) is current + vel*dt -- one step AHEAD.
            # _social_ped_snapshot removes one velocity step to recover the exact
            # current frame (NOT the [...,0] stale frame, NOT the ODE-step index j).
            # See tests/social/test_cfm_social_snapshot.py.
            if obs_positions.dim() == 3:
                ped_pos_now, ped_vel_now = _social_ped_snapshot(
                    obs_positions, obs_velocities, control_history, config.dt
                )
            else:
                ped_pos_now = obs_positions
                ped_vel_now = obs_velocities
            if name == "legibility":
                goal_dir = goal_pos.squeeze(0) - start_pos.squeeze(0)
                grad = social_grad_fns[name](
                    x_1_pred, ped_pos_now, ped_vel_now, goal_dir
                )
            elif name == "group":
                # group takes ped positions + group_pairs (int indices), NOT velocities.
                # Skip if no groups configured (avoids None subscript crash).
                if config.group_pairs is None or config.group_pairs.numel() == 0:
                    continue
                grad = social_grad_fns[name](x_1_pred, ped_pos_now, config.group_pairs)
            else:
                grad = social_grad_fns[name](x_1_pred, ped_pos_now, ped_vel_now)
            grad_norm = torch.norm(grad, keepdim=True)
            normalized = grad * u_norm / (grad_norm + 1e-8)
            if apply_markup:
                normalized = normalized * markup
            if control_history is not None:
                # parity with goal/cbf: don't let social grads update committed
                # history slots (mask defined above when control_history is not None)
                normalized = normalized * mask
            coef = getattr(config, f"{name}_margin_coef")
            contribution = coef * normalized
            social_terms.append(contribution)
            if sink is not None:
                sink.add_grad(j, name, contribution)

        goal_contribution = config.goal_margin_coef * normalized_grad_goal
        cbf_contribution = safe_coef * normalized_grad_cbf * markup
        if sink is not None:
            sink.add_grad(j, "goal", goal_contribution)
            sink.add_grad(j, "cbf", cbf_contribution)
        u_t_pred_new = (
            u_t_pred
            + goal_contribution
            + cbf_contribution
            + sum(
                social_terms, torch.zeros_like(u_t_pred)
            )  # typed init: empty sum stays tensor
        )
        noisy_action_seq = (
            noisy_action_seq
            + (t_next.reshape(-1, 1, 1) - noise_level.reshape(-1, 1, 1)) * u_t_pred_new
        )
        noise_level = t_next

    return noisy_action_seq * config.space_scale


def synthesize_control(
    model,
    mppi_solver,
    config: CFMConfig,
    ego_state,
    goal_pos,
    noisy_action_seq,
    noise_level,
    obs_positions,
    obs_velocities,
    planning_horizon,
    histories: dict = None,
    recorder=None,
    sink=None,
    **mppi_kwargs,
):
    """
    Synthesizes control for the ego vehicle using CFM.
    ego_state: [1, dim_state]
    goal_pos: [1, 2]
    obs_positions: [1, num_obst, 2]
    obs_velocities: [1, num_obst, 2]
    planning_horizon: scalar
    """
    if histories is None:
        histories = {}

    state_hist = histories.get("ego_state")
    control_hist_sin = histories.get("ego_control_sin")
    obs_state_hist = histories.get("obs_state")
    obs_control_hist = histories.get("obs_control")

    # Extract data tensors from history wrapper objects
    state_history = state_hist.get() if state_hist else None
    control_history_sin = control_hist_sin.get() if control_hist_sin else None
    obstacle_state_history = obs_state_hist.get() if obs_state_hist else None
    obstacle_control_history = obs_control_hist.get() if obs_control_hist else None

    history_len = len(state_hist) if state_hist else 0

    if control_history_sin is not None:
        vel_obs_seq = obs_velocities.unsqueeze(-1).repeat(
            1, 1, 1, planning_horizon - history_len
        )
        pos_obs_seq = obs_positions.unsqueeze(-1) + torch.cumsum(
            vel_obs_seq * config.dt, dim=3
        )
        pos_obs_seq = torch.cat([obstacle_state_history, pos_obs_seq], dim=3)
        vel_obs_seq = torch.cat([obstacle_control_history, vel_obs_seq], dim=3)
        goal_cfm = goal_pos - state_history[:, :2, 0]
        pos_obs_seq_cfm = pos_obs_seq - state_history[:, :2, 0].unsqueeze(-1)
    else:
        vel_obs_seq = obs_velocities.unsqueeze(-1).repeat(1, 1, 1, planning_horizon)
        pos_obs_seq = obs_positions.unsqueeze(-1) + torch.cumsum(
            vel_obs_seq * config.dt, dim=3
        )
        goal_cfm = goal_pos - ego_state[:, :2]
        pos_obs_seq_cfm = pos_obs_seq - ego_state[:, :2].unsqueeze(1).unsqueeze(-1)

    # [n_samples, 2, planning_horizon]
    if recorder is not None:
        recorder.start_section("cfm")
    controls_sin = run_CFM(
        model,
        config,
        noisy_action_seq,
        noise_level,
        torch.zeros(1, 2, device=config.device),
        goal_cfm,
        pos_obs_seq_cfm,
        vel_obs_seq,
        control_history_sin,
        sink=sink,
    ).detach()
    if recorder is not None:
        recorder.end_section("cfm")
    if sink is not None:
        # controls_sin is the CFM generative-prior proposal set [S,2,H]; the proposal
        # source for §4.10, NOT the post-MPPI elite (x_sin). history_len is the
        # valid-future boundary on H.
        sink.add_step(controls_sin, history_len)

    if recorder is not None:
        recorder.start_section("mppi")
    with torch.no_grad():
        x_dyn, x_sin = mppi_solver.forward(
            ego_state,
            controls_sin[:, :, history_len:].transpose(1, 2),
            planning_horizon - history_len,
            goal_pos,
            pos_obs_seq[:, :, :, history_len:].squeeze(0).transpose(1, 2),
            config.agent_radius,
            **mppi_kwargs,
        )
        x_dyn = x_dyn.unsqueeze(0).transpose(
            1, 2
        )  # [1, 2, planning_horizon-history_len]
        x_sin = x_sin.unsqueeze(0).transpose(
            1, 2
        )  # [1, 2, planning_horizon-history_len]
        x_sin = (
            torch.cat([control_history_sin, x_sin], dim=2)
            if control_history_sin is not None
            else x_sin
        )
    if recorder is not None:
        recorder.end_section("mppi")

    return x_dyn, x_sin
