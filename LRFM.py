
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal

class SubNet(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.norm(x))
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = F.relu(self.linear_3(x))
        return x

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

# Backward-compatible alias if users prefer the original class name.
LRFM = LRFM_CT_MRI

# if __name__ == "__main__":
#     # quick sanity test
#     B = 8
#     model = LRFM_CT_MRI(input_dims=(64, 128),
#                         hidden_dims=(32, 64),
#                         dropouts=(0.2, 0.2, 0.2),
#                         output_dim=5,
#                         rank=4,
#                         use_softmax=False)
#     ct = torch.randn(B, 64)
#     mri = torch.randn(B, 128)
#     y = model(ct, mri)
#     print("Output:", y.shape)
