"""Advanced PINN-MPC architectures, compared against Kardamaki's 64-16-16 baseline.

Kardamaki et al. 2026 used a small fully-connected MLP (3 hidden layers,
widths [64, 16, 16] = 1,794 trainable params). Literature offers several
alternatives we test:

  1. **Wider MLP** (3 x 128, ~50k params) - more capacity for the policy
  2. **Deeper MLP** (5 x 64, ~17k params) - depth at fixed width
  3. **SIREN-style** (sinusoidal activations) - good for PINN problems
     [Sitzmann et al. NeurIPS 2020]
  4. **Adaptive Tanh** with learnable scaling - smoother gradients near saturation
  5. **Skip-connection MLP** (ResNet-style) - easier optimization

All keep the same input dim (6) and output dim (2) so the loss + training
code in `pinn_siso.py` works unchanged.

References:
  [1] Kardamaki et al. (2026). J. Process Control 158, 103634. - baseline 64-16-16 with tanh.
  [2] Sitzmann, V., et al. (2020). "Implicit Neural Representations
      with Periodic Activation Functions (SIREN)." NeurIPS 2020.
  [3] Wang, S., Yu, X., Perdikaris, P. (2022). "When and why PINNs fail
      to train: A neural tangent kernel perspective." J. Comput. Phys.
      449, 110768. - architecture impact on PINN trainability.
  [4] He, K., et al. (2016). "Deep Residual Learning." CVPR. - skip connections.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .pinn_siso import DEVICE, PINN_Controller


# ---------------------------------------------------------------------------
# Variant 1: Wider / Deeper MLP (parameter scaling)
# ---------------------------------------------------------------------------

def make_wider_pinn(K_VALVE: float = 0.7, A: float = 1.0) -> PINN_Controller:
    """3 hidden layers x 128 units (~50k params, ~28x Kardamaki)."""
    return PINN_Controller(hidden_layers=[128, 128, 128],
                            K_VALVE=K_VALVE, A=A).to(DEVICE)


def make_deeper_pinn(K_VALVE: float = 0.7, A: float = 1.0) -> PINN_Controller:
    """5 hidden layers x 64 units (~17k params, ~10x Kardamaki)."""
    return PINN_Controller(hidden_layers=[64, 64, 64, 64, 64],
                            K_VALVE=K_VALVE, A=A).to(DEVICE)


# ---------------------------------------------------------------------------
# Variant 2: SIREN-style sinusoidal activations
# ---------------------------------------------------------------------------
class SineActivation(nn.Module):
    """SIREN sinusoidal activation (Sitzmann et al. 2020)."""
    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self.omega_0 * x)


class SIREN_PINN(nn.Module):
    """PINN with SIREN initialization + sinusoidal activations.

    Same I/O signature as Kardamaki's PINN_Controller. Particularly well-
    suited for PINN problems because sinusoidal nets have good high-
    frequency representation (their Sec 5 / Theorem 1).
    """
    def __init__(self, hidden_layers: list[int] = [64, 16, 16],
                 K_VALVE: float = 0.7, A: float = 1.0,
                 omega_0: float = 30.0):
        super().__init__()
        self.x_min, self.x_max = 0.0, 4.0
        self.u_min, self.u_max = 0.0, 1.0
        self.du_max = 0.2
        self.K_VALVE = K_VALVE
        self.A = A
        layers = []
        in_dim = 6
        for i, width in enumerate(hidden_layers):
            lin = nn.Linear(in_dim, width)
            # SIREN initialization (Sitzmann et al. 2020 Sec 3.2)
            with torch.no_grad():
                if i == 0:
                    lin.weight.uniform_(-1.0 / in_dim, 1.0 / in_dim)
                else:
                    bound = math.sqrt(6.0 / in_dim) / omega_0
                    lin.weight.uniform_(-bound, bound)
            layers += [lin, SineActivation(omega_0)]
            in_dim = width
        layers += [nn.Linear(in_dim, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, t, x0, u0, ysp, d0):
        net_in = torch.stack((t, x0, u0, ysp, d0, x0 - ysp), dim=-1)
        x_pred, u_pred = self.net(net_in).unbind(-1)
        return x_pred, u_pred

    # Reuse Kardamaki's loss verbatim (the loss is architecture-agnostic)
    loss = PINN_Controller.loss


def make_siren_pinn(K_VALVE: float = 0.7, A: float = 1.0,
                     hidden=[64, 16, 16],
                     omega_0: float = 30.0) -> SIREN_PINN:
    return SIREN_PINN(hidden_layers=hidden, K_VALVE=K_VALVE, A=A,
                       omega_0=omega_0).to(DEVICE)


# ---------------------------------------------------------------------------
# Variant 3: Skip-connection ("ResNet") PINN
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x):
        return x + torch.tanh(self.lin2(torch.tanh(self.lin1(x))))


class ResNet_PINN(nn.Module):
    """PINN with residual blocks. Easier optimization for deeper nets
    [He et al. 2016 CVPR]."""
    def __init__(self, hidden: int = 64, n_blocks: int = 3,
                 K_VALVE: float = 0.7, A: float = 1.0):
        super().__init__()
        self.x_min, self.x_max = 0.0, 4.0
        self.u_min, self.u_max = 0.0, 1.0
        self.du_max = 0.2
        self.K_VALVE = K_VALVE
        self.A = A
        self.input_proj = nn.Linear(6, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden)
                                       for _ in range(n_blocks)])
        self.output = nn.Linear(hidden, 2)

    def forward(self, t, x0, u0, ysp, d0):
        net_in = torch.stack((t, x0, u0, ysp, d0, x0 - ysp), dim=-1)
        h = torch.tanh(self.input_proj(net_in))
        for block in self.blocks:
            h = block(h)
        x_pred, u_pred = self.output(h).unbind(-1)
        return x_pred, u_pred

    loss = PINN_Controller.loss


def make_resnet_pinn(hidden: int = 64, n_blocks: int = 3,
                      K_VALVE: float = 0.7, A: float = 1.0) -> ResNet_PINN:
    return ResNet_PINN(hidden=hidden, n_blocks=n_blocks,
                        K_VALVE=K_VALVE, A=A).to(DEVICE)


# ---------------------------------------------------------------------------
# Architecture registry (for the bench)
# ---------------------------------------------------------------------------
ARCH_REGISTRY = {
    "kardamaki_64_16_16":  lambda: PINN_Controller(
        hidden_layers=[64, 16, 16]).to(DEVICE),                   # baseline
    "wider_3x128":         lambda: make_wider_pinn(),             # 28x params
    "deeper_5x64":         lambda: make_deeper_pinn(),            # 10x params
    "siren_3x64":          lambda: make_siren_pinn(hidden=[64, 64, 64]),
    "siren_kardamaki":     lambda: make_siren_pinn(
        hidden=[64, 16, 16]),                                      # SIREN at K's size
    "resnet_64x3":         lambda: make_resnet_pinn(hidden=64, n_blocks=3),
    "resnet_128x3":        lambda: make_resnet_pinn(hidden=128, n_blocks=3),
}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== Architecture comparison ===\n")
    print(f"  {'name':<22}{'params':>10}  description")
    print(f"  " + "-" * 70)
    for name, fn in ARCH_REGISTRY.items():
        m = fn()
        n = count_params(m)
        desc = ""
        if "kardamaki_64_16_16" == name: desc = "baseline (the paper)"
        elif "wider" in name: desc = "more width = more policy capacity"
        elif "deeper" in name: desc = "more depth at fixed width"
        elif "siren" in name: desc = "sinusoidal activations (Sitzmann 2020)"
        elif "resnet" in name: desc = "skip connections (He 2016)"
        print(f"  {name:<22}{n:>10}  {desc}")
