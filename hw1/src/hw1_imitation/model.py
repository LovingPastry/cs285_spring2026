"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn

class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        in_dim = state_dim      # 输入层
        for h in hidden_dims:   # 隐藏层
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim * chunk_size))

        self.net = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss between predicted and target action chunks."""
        pred_chunk = self.sample_actions(state)
        loss = torch.mean((pred_chunk - action_chunk) ** 2)
        return loss


    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Generate action chunks using MLP."""
        batch_size = state.shape[0]
        output = self.net(state)
        # Reshape to (batch, chunk_size, action_dim)
        return output.view(batch_size, self.chunk_size, self.action_dim)




class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
        )
        layers = []
        in_dim = state_dim + action_dim*chunk_size + 32      # 输入层
        for h in hidden_dims:   # 隐藏层
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim * chunk_size))

        self.net = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        device = action_chunk.device

        # Sample noise and time; match action_chunk shape: (B, chunk, action_dim)
        noise = torch.randn_like(action_chunk)
        time_dist = torch.distributions.Beta(1.5, 1.0)  # 保持与pi0最佳实践相同
        t = time_dist.sample((batch_size, 1)).to(device) * 0.999 + 0.001

        t_exp = t.view(batch_size, 1, 1)  # broadcast over chunk/action dims
        x_t = t_exp *action_chunk  + (1 - t_exp) * noise
        u_t = action_chunk - noise

        x_t_flat = x_t.reshape(batch_size, -1)
        t_feat = self.time_embed(t)
        net_in = torch.cat((state, x_t_flat, t_feat), dim=1)
        v_t = self.net(net_in).reshape_as(action_chunk)

        loss = torch.mean((v_t - u_t) ** 2)
        return loss


    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        device = state.device
        dt = 1 / float(num_steps)
        x_t = torch.randn(
            batch_size, self.chunk_size, self.action_dim, device=device
        )

        for i in range(num_steps):
            t = torch.full((batch_size, 1), i * dt, device=device)
            x_t_flat = x_t.reshape(batch_size, -1)
            net_in = torch.cat(
                (state, x_t_flat, self.time_embed(t)), 
                dim=1
            )
            v_t = self.net(net_in).reshape_as(x_t)
            x_t = x_t + dt * v_t
        
        return x_t


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
