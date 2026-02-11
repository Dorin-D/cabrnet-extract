import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdyn.core import NeuralODE

from modules import DecoderBlock, EncoderBlock



class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=128, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FlowMatcher(nn.Module):
    def __init__(
            self, 
            upsample_dim: int = 64,
            hidden_dim: int = 64,
            output_dim: int = 3, 
            *, 
            strides: list[int] = [1, 2], 
            n_convs: list[int] = [5, 5],
            sigma_min: float = 1e-4,
        ):
        super(FlowMatcher, self).__init__()

        self.sigma_min = sigma_min

        self.time_emb = GaussianFourierProjection(hidden_dim//2)

        self.init_proj = nn.Conv2d(upsample_dim+output_dim,
                                   hidden_dim,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0)

        self.encoder = nn.ModuleList()
        for i in range(len(n_convs)):
            self.encoder.append(
                EncoderBlock(
                    n_in=hidden_dim, 
                    n_out=hidden_dim, 
                    down_kernel_size=7,
                    stride=strides[i],
                    n_convs=n_convs[i]
                )
            )  

        self.decoder = nn.ModuleList()
        for i in reversed(range(len(n_convs))):
            self.decoder.append(
                DecoderBlock(
                    n_in=hidden_dim,
                    n_out=hidden_dim,
                    stride=strides[i],
                    up_kernel_size=7,
                    n_convs=n_convs[i]
                )
            )

        self.out_linear = nn.Conv2d(hidden_dim,
                                    output_dim,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        
    def forward(self, x, y, t):
        # x (image) (B, output_dim, 224, 224)
        # y (upsampled features) (B, upsample_dim, 224, 224)
        # t (B, )
        B, _, H, W = x.shape
        temb = self.time_emb(t) # (B, 256)

        x = torch.cat([x, y], dim=1) # (B, upsample_dim+output_dim, H, W))
        x = self.init_proj(x) # (B, hidden_dim, H, W)

        hidden_states = []
        for enc_block in self.encoder:
            x = enc_block(x+temb[:, :, None, None]) 
            hidden_states.append(x)

        for dec_block in self.decoder:
            hs = hidden_states.pop()
            x = dec_block(x+hs+temb[:, :, None, None])

        return self.out_linear(x)
    
    def loss(self, images, features):
        t = torch.rand(images.shape[0], device=images.device)

        x_0 = torch.randn_like(images)
        x_t = t[:, None, None, None] * images + (1 - (1-self.sigma_min) * t[:, None, None, None]) * x_0
        u_t = images - (1-self.sigma_min) * x_0

        v_t = self(x_t, features, t)
        loss = F.mse_loss(v_t, u_t)

        return loss
    
    def sample_image(self, features: torch.Tensor, *, steps=10):
        def solver_fn(t, Xt, *args, **kwargs):
            return self(Xt, features, t.unsqueeze(0))
        
        dummy = nn.Parameter(torch.empty(0), requires_grad=False)
        B, _, H, W = features.shape
        with torch.no_grad():
            neural_od = NeuralODE(solver_fn, 
                                  solver='midpoint', 
                                  sensitivity='adjoint',
                                  optimizable_params=[dummy])
            initial_state = torch.randn((B, self.output_dim, H, W), device=features.device)
            t_span = torch.linspace(0, 1, steps+1, device=features.device)
            traj = neural_od.trajectory(initial_state, t_span=t_span)

        return traj[-1]