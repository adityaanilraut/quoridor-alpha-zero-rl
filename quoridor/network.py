"""Policy + value network and a thin inference/training wrapper."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import NUM_PLANES, encode_state
from .game import action_space_size


def resolve_device(name="auto"):
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class QuoridorNet(nn.Module):
    def __init__(self, board_size, channels=64, res_blocks=5):
        super().__init__()
        self.board_size = board_size
        self.action_size = action_space_size(board_size)

        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(res_blocks)])

        # policy head
        self.p_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(32)
        self.p_fc = nn.Linear(32 * board_size * board_size, self.action_size)

        # value head
        self.v_conv = nn.Conv2d(channels, 3, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(3)
        self.v_fc1 = nn.Linear(3 * board_size * board_size, channels)
        self.v_fc2 = nn.Linear(channels, 1)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)

        p = F.relu(self.p_bn(self.p_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.p_fc(p)

        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))
        return p, v


class NeuralNet:
    """Wraps a QuoridorNet with inference and training helpers."""

    def __init__(self, config, device=None):
        self.config = config
        self.device = resolve_device(device or config.device)
        # Avoid CPU thread contention with MPS or single-core CPU workloads
        if self.device.type == "cpu":
            torch.set_num_threads(1)

        self.net = QuoridorNet(
            board_size=config.board_size,
            channels=config.channels,
            res_blocks=config.res_blocks,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    # ---- inference ----
    @torch.no_grad()
    def predict(self, planes):
        """Single state. Returns (policy_logits: np[A], value: float)."""
        self.net.eval()
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        logits, value = self.net(x)
        return logits[0].cpu().numpy(), float(value.item())

    @torch.no_grad()
    def predict_states(self, states):
        """Batched inference over a list of states.

        Returns (logits: np[L, A], values: np[L]) where L = len(states).
        """
        self.net.eval()
        x = torch.from_numpy(np.stack([encode_state(s) for s in states])).to(self.device)
        logits, value = self.net(x)
        return logits.cpu().numpy(), value.cpu().numpy()

    # ---- training ----
    def train_step(self, planes, target_pi, target_z):
        self.net.train()
        x = torch.from_numpy(planes).to(self.device)
        pi = torch.from_numpy(target_pi).to(self.device)
        z = torch.from_numpy(target_z).to(self.device)

        logits, value = self.net(x)
        logp = F.log_softmax(logits, dim=1)
        policy_loss = -(pi * logp).sum(dim=1).mean()
        value_loss = F.mse_loss(value.squeeze(-1), z)
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item()), float(policy_loss.item()), float(value_loss.item())

    # ---- io ----
    def save(self, path, extra=None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path, load_optimizer=True):
        ckpt = torch.load(path, map_location=self.device)
        saved_cfg = ckpt.get("config")
        if saved_cfg is not None:
            saved_bs = saved_cfg.get("board_size")
            if saved_bs is not None and saved_bs != self.config.board_size:
                raise ValueError(
                    f"checkpoint {path} was trained on board_size={saved_bs} "
                    f"but the current config uses board_size="
                    f"{self.config.board_size}. Pass the matching --preset "
                    f"(e.g. 'standard' for 9x9, 'fast' for 5x5)."
                )
        self.net.load_state_dict(ckpt["model"])
        if load_optimizer and "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                pass
        return ckpt

    def clone(self):
        other = NeuralNet(self.config, self.device)
        other.net.load_state_dict(self.net.state_dict())
        # Preserve optimizer moments so training continuity isn't lost when a
        # candidate is cloned (e.g. reverted to best after a failed gate).
        other.optimizer.load_state_dict(self.optimizer.state_dict())
        return other
