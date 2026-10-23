"""Residual-bandit collocation sampler built on :class:`RLAgent`.

The domain ``[x_lo, x_hi] × [t_lo, t_hi]`` is split into a grid of cells;
each cell is one action. Every ``probe_every``-th call to
:meth:`ResidualBanditSampler.sample` (by default, every call)

1. probes a few random cells with a handful of residual evaluations and
   stores ``(state, cell, reward)`` transitions in the agent's replay buffer,
   where the reward is the cell's mean ``|residual|`` divided by the mean
   over the probed cells (scale-free, so it stays comparable as training
   drives residuals down) and the state is the 4-vector of ``log10`` mean,
   max and std of the probed ``|residual|`` plus training progress;
2. trains the DQN on replayed transitions — with ``gamma = 0`` the Bellman
   target is the reward itself, so ``Q(s, cell)`` regresses onto the cell's
   relative residual (a contextual bandit).

Every call then draws ``policy_fraction`` of the points from cells chosen by
the ε-mixed ``softmax(Q / temperature)`` policy and the rest uniformly over
the whole domain, so no region is ever starved.

The design was prototyped in ``notebooks/03_adaptive_sampling.ipynb`` (then x-only cells)
extended from x-only cells to an ``(x, t)`` grid.
"""

from __future__ import annotations

import torch

from pinnrl.rl.rl_agent import RLAgent


class ResidualBanditSampler:
    """Collocation sampler whose cell preferences are learned by a DQN bandit.

    Args:
        domain: ``(x_lo, x_hi)`` spatial bounds (1-D PDEs only).
        time_domain: ``(t_lo, t_hi)`` temporal bounds.
        n_x_cells: Number of cells along ``x``.
        n_t_cells: Number of cells along ``t``.
        policy_fraction: Share of each batch drawn from the policy; the rest
            is uniform over the full domain.
        temperature: Softmax temperature on the Q-values.
        probe_cells: Cells probed per call to collect rewards.
        probe_points: Residual evaluations per probed cell.
        probe_every: Probe (and train the agent) on every ``probe_every``-th
            call to :meth:`sample`; other calls reuse the last state. Raise it
            when sampling every optimizer step, as the trainer does.
        agent: Optional pre-built agent; must have ``state_dim=4`` and
            ``action_dim = n_x_cells * n_t_cells``.
        device: Torch device.
    """

    STATE_DIM = 4

    def __init__(
        self,
        domain: tuple[float, float],
        time_domain: tuple[float, float],
        n_x_cells: int = 16,
        n_t_cells: int = 8,
        policy_fraction: float = 0.5,
        temperature: float = 0.5,
        probe_cells: int = 32,
        probe_points: int = 8,
        probe_every: int = 1,
        agent: RLAgent | None = None,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cpu")
        self.x_lo, self.x_hi = float(domain[0]), float(domain[1])
        self.t_lo, self.t_hi = float(time_domain[0]), float(time_domain[1])
        self.n_x_cells = n_x_cells
        self.n_t_cells = n_t_cells
        self.n_cells = n_x_cells * n_t_cells
        self.policy_fraction = policy_fraction
        self.temperature = temperature
        self.probe_cells = probe_cells
        self.probe_points = probe_points
        self.probe_every = max(1, probe_every)
        self._calls = 0
        self._state: torch.Tensor | None = None
        self.agent = agent or RLAgent(
            state_dim=self.STATE_DIM,
            action_dim=self.n_cells,
            hidden_dim=64,
            learning_rate=2e-3,
            gamma=0.0,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.998,
            memory_size=4000,
            batch_size=64,
            target_update=100,
            device=self.device,
        )
        if self.agent.action_dim != self.n_cells:
            raise ValueError(
                f"agent.action_dim={self.agent.action_dim} does not match "
                f"n_x_cells * n_t_cells = {self.n_cells}"
            )

    def _points_in_cells(self, cells: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One uniform random point inside each listed cell index."""
        dx = (self.x_hi - self.x_lo) / self.n_x_cells
        dt = (self.t_hi - self.t_lo) / self.n_t_cells
        ix = (cells // self.n_t_cells).float()
        it = (cells % self.n_t_cells).float()
        x = self.x_lo + (ix + torch.rand_like(ix)) * dx
        t = self.t_lo + (it + torch.rand_like(it)) * dt
        return x.reshape(-1, 1), t.reshape(-1, 1)

    def _uniform(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x_lo + torch.rand(n, 1, device=self.device) * (self.x_hi - self.x_lo)
        t = self.t_lo + torch.rand(n, 1, device=self.device) * (self.t_hi - self.t_lo)
        return x, t

    def probe(self, pde, model: torch.nn.Module, progress: float) -> torch.Tensor:
        """Probe random cells, push rewards to the agent, and return the state.

        ``probe_points`` random points in each of ``min(probe_cells, n_cells)``
        distinct cells are scored by ``|residual|``. The state is
        ``[log10 mean, log10 max, log10 std, progress]`` over all probed
        values (the mean is the mean of the cell means); each cell's reward
        is its mean divided by that overall mean. Each transition is stored
        with ``next_state = state``, which ``gamma = 0`` ignores.
        """
        cells = torch.randperm(self.n_cells, device=self.device)[: self.probe_cells]
        x, t = self._points_in_cells(cells.repeat_interleave(self.probe_points))
        residual = pde.compute_residual(model, x.requires_grad_(True), t.requires_grad_(True))
        res_abs = residual.detach().abs().reshape(len(cells), self.probe_points)

        cell_means = res_abs.mean(dim=1)
        overall = cell_means.mean().clamp_min(1e-12)
        state = torch.tensor(
            [
                torch.log10(overall).item(),
                torch.log10(res_abs.max().clamp_min(1e-12)).item(),
                torch.log10(res_abs.std().clamp_min(1e-12)).item(),
                float(progress),
            ],
            device=self.device,
        )
        for cell, reward in zip(cells.tolist(), (cell_means / overall).tolist(), strict=False):
            self.agent.update(state, cell, reward, state, False)
        return state

    def distribution(self, state: torch.Tensor) -> torch.Tensor:
        """ε-mixture of ``softmax(Q / temperature)`` and the uniform distribution.

        The Q-network is evaluated in eval mode (dropout off), so the same
        state always gives the same distribution.
        """
        net = self.agent.policy_net
        was_training = net.training
        net.eval()
        with torch.no_grad():
            q = net(state.unsqueeze(0)).squeeze(0)
        net.train(was_training)
        eps = float(self.agent.epsilon)
        return (1.0 - eps) * torch.softmax(q / self.temperature, dim=0) + eps / self.n_cells

    def sample(
        self, pde, model: torch.nn.Module, num_points: int, progress: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``num_points`` collocation points, as ``(x, t)`` columns.

        Probes and updates the agent first on every ``probe_every``-th call
        (and on the first); other calls reuse the last state. ``progress`` in
        [0, 1] is the training-progress entry of the state.
        """
        if self._state is None or self._calls % self.probe_every == 0:
            self._state = self.probe(pde, model, progress)
        self._calls += 1
        state = self._state
        n_policy = int(round(num_points * self.policy_fraction))
        if n_policy > 0:
            cells = torch.multinomial(self.distribution(state), n_policy, replacement=True)
        else:
            cells = torch.empty(0, dtype=torch.long, device=self.device)
        x_p, t_p = self._points_in_cells(cells)
        x_u, t_u = self._uniform(num_points - n_policy)
        return torch.cat([x_p, x_u]).detach(), torch.cat([t_p, t_u]).detach()
