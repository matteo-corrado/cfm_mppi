import torch
from cfm_mppi.reward import single_cbf_reward_fn_pairwise, single_goal_reward_fn
from cfm_mppi.social_reward import single_proxemic_reward_fn
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
    norm_yield_T_safe: float = 1.5
    norm_yield_R_conflict: float = 1.0


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
            # leg/norm_side/norm_yield/group added in later tasks
        ]
        for name, apply_markup in SOCIAL_TERM_META:
            if name not in social_grad_fns:
                continue
            # obs_positions/obs_velocities are [n_peds, 2, horizon] in the real pipeline;
            # social reward fns take a static [n_peds, 2] snapshot. Use the nearest-term
            # slice (index 0), NOT the ODE-step index j — j is a diffusion step, unrelated
            # to horizon timesteps, and j could exceed horizon (IndexError).
            ped_pos_now = (
                obs_positions[..., 0] if obs_positions.dim() == 3 else obs_positions
            )
            ped_vel_now = (
                obs_velocities[..., 0] if obs_velocities.dim() == 3 else obs_velocities
            )
            grad = social_grad_fns[name](x_1_pred, ped_pos_now, ped_vel_now)
            grad_norm = torch.norm(grad, keepdim=True)
            normalized = grad * u_norm / (grad_norm + 1e-8)
            if apply_markup:
                normalized = normalized * markup
            coef = getattr(config, f"{name}_margin_coef")
            social_terms.append(coef * normalized)

        u_t_pred_new = (
            u_t_pred
            + config.goal_margin_coef * normalized_grad_goal
            + safe_coef * normalized_grad_cbf * markup
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
    ).detach()
    if recorder is not None:
        recorder.end_section("cfm")

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
