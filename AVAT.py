
"""
AVAT -> Dynamic Path Attention (DPA)
------------------------------------
This file implements two modules that mirror the provided diagrams:
  1) AVATFrontEnd: builds Q, K, V from (q, reference feature points, stripe position bias, mu)
     - q is adjusted by a learnable offset network (theta_offset)
     - reference feature points are added to q (residual)
     - stripe position bias is bilinearly interpolated and added to q
     - k and v are produced by shallow conv stems
     - mu (Lagrange multiplier) is one-hot encoded and passed forward to DPA
  2) DynamicPathAttention: mixes two paths (Conv1, Conv3) for K and V using
     Dynamic Path Selectors (DPS) with Gumbel-Softmax (AvgPool -> LN -> ReLU -> LN -> Linear -> Gumbel)
     and applies Multi-Head Attention (MHA).

Notes
-----
* This is a faithful, runnable interpretation of the figures; specific kernel layouts
  can be customized as needed.
* All tensors are 4D BCHW. For 3D MRI, wrap per-slice or extend to 3D kernels.
"""
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F

# -------------------- small helpers --------------------
def bilinear_like(x: torch.Tensor, target: torch.Size) -> torch.Tensor:
    """Bilinear resize to target HxW (keeps batch/channels)."""
    _, _, H, W = target
    return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

class OneHot(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B,) int
        b = idx.shape[0]
        y = torch.zeros(b, self.num_classes, device=idx.device, dtype=torch.float32)
        y.scatter_(1, idx.view(-1,1), 1.0)
        return y

# -------------------- conv stems --------------------
class ConvStem(nn.Module):
    """Simple 2-layer stem used to produce K and V from q."""
    def __init__(self, c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, c//8), c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.GroupNorm(max(1, c//8), c),
        )
    def forward(self, x): return self.net(x)

class OffsetNet(nn.Module):
    """theta_offset: lightweight adjustment to q (residual)."""
    def __init__(self, c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
        )
    def forward(self, q): 
        return q + self.net(q)

# -------------------- DPS --------------------
class DynamicPathSelector(nn.Module):
    """AvgPool -> LN -> ReLU -> LN -> Linear(2) -> Gumbel-Softmax."""
    def __init__(self, channels: int, cond_dim: int = 0, tau: float = 1.0, hard: bool = False):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.ln1 = nn.LayerNorm(channels + cond_dim)
        self.ln2 = nn.LayerNorm(channels + cond_dim)
        self.fc = nn.Linear(channels + cond_dim, 2)
        self.tau = tau
        self.hard = hard

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B,C,H,W); cond: (B,cond_dim) optional
        b, c, _, _ = x.shape
        h = self.avg(x).view(b, c)               # (B,C)
        if cond is not None:
            h = torch.cat([h, cond], dim=-1)     # (B,C+cond)
        h = self.ln1(h)
        h = F.relu(h, inplace=True)
        h = self.ln2(h)
        logits = self.fc(h)                      # (B,2)
        alpha = F.gumbel_softmax(logits, tau=self.tau, hard=self.hard, dim=-1)
        return alpha                              # (B,2), sum=1

class ConvPath(nn.Module):
    """A path option: depthwise + pointwise conv followed by Softplus."""
    def __init__(self, c: int, k: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, k, padding=k//2, groups=c, bias=False),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.act = nn.Softplus()
    def forward(self, x): return self.act(self.block(x))

# -------------------- DPA --------------------
class DynamicPathAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, mu_classes: int = 0, gumbel_tau: float = 1.0, gumbel_hard: bool = False):
        super().__init__()
        self.channels = channels
        self.mu_classes = mu_classes

        # two paths for K and for V
        self.k_path1 = ConvPath(channels, 1)
        self.k_path3 = ConvPath(channels, 3)
        self.v_path1 = ConvPath(channels, 1)
        self.v_path3 = ConvPath(channels, 3)

        cond_dim = mu_classes if mu_classes > 0 else 0
        self.dps_k = DynamicPathSelector(channels, cond_dim, gumbel_tau, gumbel_hard)
        self.dps_v = DynamicPathSelector(channels, cond_dim, gumbel_tau, gumbel_hard)

        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mu_onehot: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Build K and V via two paths and DPS mixing
        k1, k3 = self.k_path1(k), self.k_path3(k)
        v1, v3 = self.v_path1(v), self.v_path3(v)
        alpha_k = self.dps_k(k, mu_onehot)
        alpha_v = self.dps_v(v, mu_onehot)
        # mix per-sample
        def mix(alpha, a, b):
            w0 = alpha[:,0][:,None,None,None]
            w1 = alpha[:,1][:,None,None,None]
            return w0*a + w1*b
        k_mix = mix(alpha_k, k1, k3)
        v_mix = mix(alpha_v, v1, v3)

        # MHA over tokens
        B,C,H,W = q.shape
        def tok(x): return x.flatten(2).transpose(1,2)  # (B,HW,C)
        out,_ = self.mha(tok(q), tok(k_mix), tok(v_mix))
        out = out.transpose(1,2).view(B,C,H,W)
        return self.out(out)

# -------------------- AVAT front-end --------------------
class AVATFrontEnd(nn.Module):
    """Implements the left part of the AVAT figure to produce (q_adj, k, v, mu_onehot)."""
    def __init__(self, channels: int, mu_classes: int = 0):
        super().__init__()
        self.theta_offset = OffsetNet(channels)
        self.k_stem = ConvStem(channels)
        self.v_stem = ConvStem(channels)
        self.mu_classes = mu_classes
        self.mu_encoder = OneHot(mu_classes) if mu_classes > 0 else None

    def forward(self, q: torch.Tensor, ref_points: Optional[torch.Tensor] = None,
                stripe_bias: Optional[torch.Tensor] = None, mu: Optional[torch.Tensor] = None):
        # q adjustment by theta_offset
        q_adj = self.theta_offset(q)
        # add reference points (residual)
        if ref_points is not None:
            q_adj = q_adj + ref_points
        # add stripe position bias (bilinear interpolation to q size)
        if stripe_bias is not None:
            if stripe_bias.shape[-2:] != q_adj.shape[-2:]:
                stripe_bias = F.interpolate(stripe_bias, size=q_adj.shape[-2:], mode='bilinear', align_corners=False)
            q_adj = q_adj + stripe_bias
        # stems for K/V
        k = self.k_stem(q_adj)
        v = self.v_stem(q_adj)
        mu_onehot = self.mu_encoder(mu) if (self.mu_encoder is not None and mu is not None) else None
        return q_adj, k, v, mu_onehot

# -------------------- Full AVAT+DPA block --------------------
class AVAT_DPA_Block(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, mu_classes: int = 0, gumbel_tau: float = 1.0, gumbel_hard: bool = False):
        super().__init__()
        self.front = AVATFrontEnd(channels, mu_classes)
        self.dpa = DynamicPathAttention(channels, num_heads, mu_classes, gumbel_tau, gumbel_hard)

    def forward(self, q, ref_points=None, stripe_bias=None, mu=None):
        q_adj, k, v, mu_onehot = self.front(q, ref_points, stripe_bias, mu)
        return self.dpa(q_adj, k, v, mu_onehot)

# -------------------- example --------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B,C,H,W = 2, 64, 32, 32
    q = torch.randn(B,C,H,W)
    ref = torch.randn(B,C,H,W) * 0.1
    bias = torch.randn(B,C,8,8) * 0.05  # low-res stripe bias (will be bilinearly upsampled)
    mu = torch.tensor([0,1])  # two classes of Lagrange multiplier for demo
    model = AVAT_DPA_Block(channels=C, num_heads=8, mu_classes=2, gumbel_tau=1.0, gumbel_hard=False)
    y = model(q, ref_points=ref, stripe_bias=bias, mu=mu)
    print("Output:", y.shape)
