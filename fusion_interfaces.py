
from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class REncoder(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 64):
        super().__init__()
        self.local = nn.Sequential(
            ConvBNAct(in_channels, base_channels, 3, 1),
            ConvBNAct(base_channels, base_channels, 3, 1),
        )
        self.global_ = nn.Sequential(
            ConvBNAct(in_channels, base_channels, 3, 2),
            ConvBNAct(base_channels, base_channels*2, 3, 2)
        )
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f_local = self.local(x)
        f_global = self.global_(x)
        return {'global': f_global, 'local': f_local}

class GFAB(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(c_in, c_out, 3, 1),
            ConvBNAct(c_out, c_out, 3, 1),
        )
    def forward(self, f_ct: torch.Tensor, f_mri: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.block(f_ct), self.block(f_mri)

class LFAB(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(c_in, c_out, 3, 1),
            ConvBNAct(c_out, c_out, 3, 1),
        )
    def forward(self, f_ct: torch.Tensor, f_mri: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.block(f_ct), self.block(f_mri)


class LRFM_CT_MRI(nn.Module):
    """
    Low-Rank Fusion Model for **two modalities** (CT & MRI).
    - Low-rank fusion uses two factors (CT, MRI).
    Args:
        input_dims : (ct_in, mri_in)
        hidden_dims: (ct_hidden, mri_hidden)
        dropouts   : (ct_prob, mri_prob, post_fusion_prob)
        output_dim : number of output targets
        rank       : low-rank factorization rank (>=1)
        use_softmax: whether to apply softmax to output logits
    """
    def __init__(
        self,
        input_dims: tuple[int, int],
        hidden_dims: tuple[int, int],
        dropouts: tuple[float, float, float],
        output_dim: int,
        rank: int,
        use_softmax: bool = False,
    ):
        super().__init__()
        # inputs / hiddens
        self.ct_in, self.mri_in = input_dims
        self.ct_hidden, self.mri_hidden = hidden_dims
        self.ct_prob, self.mri_prob, self.post_fusion_prob = dropouts
        self.output_dim = output_dim
        self.rank = rank
        self.use_softmax = use_softmax

        # pre-fusion subnetworks
        self.ct_subnet = SubNet(self.ct_in, self.ct_hidden, self.ct_prob)
        self.mri_subnet = SubNet(self.mri_in, self.mri_hidden, self.mri_prob)

        # low-rank fusion parameters
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.ct_factor = Parameter(torch.Tensor(self.rank, self.ct_hidden + 1, self.output_dim))
        self.mri_factor = Parameter(torch.Tensor(self.rank, self.mri_hidden + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        # init
        xavier_normal(self.ct_factor)
        xavier_normal(self.mri_factor)
        xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0.0)

    def forward(self, ct_x: torch.Tensor, mri_x: torch.Tensor) -> torch.Tensor:
        """
        ct_x  : (B, ct_in)
        mri_x : (B, mri_in)
        returns logits of shape (B, output_dim)
        """
        ct_h = self.ct_subnet(ct_x)     # (B, ct_hidden)
        mri_h = self.mri_subnet(mri_x)  # (B, mri_hidden)
        B = ct_h.shape[0]

        # augment with 1 for bias trick
        if ct_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _ct_h  = torch.cat((Variable(torch.ones(B, 1).type(DTYPE), requires_grad=False), ct_h), dim=1)
        _mri_h = torch.cat((Variable(torch.ones(B, 1).type(DTYPE), requires_grad=False), mri_h), dim=1)

        # factor projections
        fusion_ct  = torch.matmul(_ct_h, self.ct_factor)     # (B, rank, out)
        fusion_mri = torch.matmul(_mri_h, self.mri_factor)    # (B, rank, out)

        # elementwise mix (rank-wise)
        fusion_zy = fusion_ct * fusion_mri                    # (B, rank, out)

        # weighted sum across rank + bias
        out = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        out = out.view(-1, self.output_dim)
        out = self.post_fusion_dropout(out)
        if self.use_softmax:
            out = F.softmax(out, dim=-1)
        return out

class RDecoder(nn.Module):
    def __init__(self, c_global: int, c_local: int, c_out: int = 1):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(c_global, c_local, kernel_size=4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(c_local),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            ConvBNAct(c_local*2, c_local, 3, 1),
            nn.Conv2d(c_local, c_out, 1)
        )
    def forward(self, g_fused: torch.Tensor, l_fused: torch.Tensor) -> torch.Tensor:
        g_up = self.up(g_fused)
        x = torch.cat([g_up, l_fused], dim=1)
        return self.fuse(x)

class FusionProcess(nn.Module):
    def __init__(self, base_channels: int = 64, rank: int = 4, aux_out_channels: int = 1):
        super().__init__()
        self.encoder = REncoder(in_channels=1, base_channels=base_channels)
        cg = base_channels * 2
        cl = base_channels
        self.gfab = GFAB(cg, cg)
        self.lfab = LFAB(cl, cl)
        self.lrfm_g = LRFM(c_ct=cg,  c_mri=cg,  c_out=cg, rank=rank)
        self.lrfm_l = LRFM(c_ct=cl,  c_mri=cl,  c_out=cl, rank=rank)
        self.decoder = RDecoder(c_global=cg, c_local=cl, c_out=aux_out_channels)
    def forward(self, ct_img: torch.Tensor, mri_img: torch.Tensor):
        ct_feats  = self.encoder(ct_img)
        mri_feats = self.encoder(mri_img)
        ct_g, mri_g = self.gfab(ct_feats['global'], mri_feats['global'])
        ct_l, mri_l = self.lfab(ct_feats['local'],  mri_feats['local'])
        g_fused = self.lrfm_g(ct_g, mri_g)
        l_fused = self.lrfm_l(ct_l, mri_l)
        aux = self.decoder(g_fused, l_fused)
        return {'g_fused': g_fused, 'l_fused': l_fused, 'auxiliary': aux}

