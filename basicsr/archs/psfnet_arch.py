import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY

class PFA(nn.Module):

    def __init__(self, dim, n_iter=3, init_grid_scale=8, var_scale_init=0.1, beta_init=10.0):
        super().__init__()
        self.n_iter = n_iter
        self.init_grid_scale = init_grid_scale
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))
        self.var_scale = nn.Parameter(torch.tensor(var_scale_init, dtype=torch.float32))

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        g_h = max(1, H // self.init_grid_scale)
        g_w = max(1, W // self.init_grid_scale)
        M = g_h * g_w
        prototype_mean = F.adaptive_avg_pool2d(x, (g_h, g_w))
        prototype_ex2 = F.adaptive_avg_pool2d(x * x, (g_h, g_w))
        prototype_var = torch.clamp(prototype_ex2 - prototype_mean * prototype_mean, min=0.0)
        prototype_mean = rearrange(prototype_mean, 'b c gh gw -> b (gh gw) c')
        prototype_var = rearrange(prototype_var, 'b c gh gw -> b (gh gw) c')
        var_norm = prototype_var / (prototype_var.mean(dim=1, keepdim=True) + 1e-06)
        prototype_centroid = prototype_mean + self.var_scale * var_norm
        for _ in range(self.n_iter):
            dist = torch.cdist(x_flat, prototype_centroid)
            affinity = F.softmax(-self.beta * dist, dim=-1)
            denom = affinity.sum(dim=1, keepdim=True).transpose(1, 2) + 1e-08
            prototype_centroid = torch.bmm(affinity.transpose(1, 2), x_flat) / denom
        return (affinity, prototype_centroid)

class PGCA(nn.Module):

    def __init__(self, dim, qk_dim=24, num_heads=1, tau_base=1.0, var_alpha=1.0, eps=1e-06):
        super().__init__()
        assert qk_dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = qk_dim // num_heads
        self.scale = (self.head_dim + 1e-08) ** (-0.5)
        self.eps = eps
        self.q_proj = nn.Conv2d(dim, qk_dim, 1)
        self.k_proj = nn.Conv2d(dim, qk_dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)
        self.out_proj = nn.Conv2d(dim, dim, 1)
        self.k_pos = nn.Conv2d(qk_dim, qk_dim, 3, 1, 1, groups=qk_dim)
        self.pre_norm_q = nn.GroupNorm(1, qk_dim)
        self.pre_norm_k = nn.GroupNorm(1, qk_dim)
        self.pre_norm_v = nn.GroupNorm(1, dim)
        self.res_gate = nn.Parameter(torch.tensor(1.0))
        self.tau_base = nn.Parameter(torch.tensor(float(tau_base)))
        self.var_alpha = nn.Parameter(torch.tensor(float(var_alpha)))

    @torch.no_grad()
    def _prototype_variance(self, x_flat, affinity, prototype_feat):
        denom = affinity.sum(dim=1, keepdim=True).transpose(1, 2) + self.eps
        x2 = x_flat * x_flat
        m2 = torch.bmm(affinity.transpose(1, 2), x2) / denom
        var = torch.clamp(m2 - prototype_feat * prototype_feat, min=0.0)
        var_norm = var.mean(dim=-1, keepdim=True)
        var_norm = var_norm / (var_norm.mean(dim=1, keepdim=True) + self.eps)
        return var_norm

    def forward(self, x, affinity, prototype_feat):
        B, C, H, W = x.shape
        N = H * W
        prototype_map = rearrange(prototype_feat, 'b m c -> b c m 1')
        q = self.q_proj(prototype_map)
        q = self.pre_norm_q(q)
        q = rearrange(q, 'b (h d) m 1 -> b h m d', h=self.num_heads)
        k = self.k_proj(x)
        v = self.v_proj(x)
        k = self.pre_norm_k(k)
        v = self.pre_norm_v(v)
        k = k + self.k_pos(k)
        k = rearrange(k, 'b (h d) h1 w1 -> b h (h1 w1) d', h=self.num_heads)
        v = rearrange(v, 'b (h d) h1 w1 -> b h (h1 w1) d', h=self.num_heads, d=C // self.num_heads)
        prior = affinity.transpose(1, 2)
        prior_log = (prior + self.eps).log().unsqueeze(1)
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        with torch.no_grad():
            var_norm = self._prototype_variance(x_flat, affinity, prototype_feat)
        tau = self.tau_base * (1.0 + self.var_alpha * var_norm)
        tau = tau.unsqueeze(1)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        logits = logits / (tau + self.eps)
        logits = logits + prior_log
        attn = F.softmax(logits, dim=-1)
        prototype_out = torch.matmul(attn, v)
        prototype_out = rearrange(prototype_out, 'b h m d -> b (h d) m 1')
        x_out = torch.matmul(affinity, rearrange(prototype_out, 'b c m 1 -> b m c'))
        x_out = rearrange(x_out, 'b (h w) c -> b c h w', h=H, w=W)
        y = self.out_proj(x_out)
        return x + self.res_gate * y

class SFCM(nn.Module):

    def __init__(self, dim, down=2):
        super().__init__()
        assert down in [1, 2, 4], f'SFCM: down 仅支持 1/2/4，当前输入为 {down}'
        self.dim = dim
        self.down = down
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.conv1 = nn.Conv2d(dim, 2 * dim, 1)
        self.conv2 = nn.Conv2d(2 * dim, dim, 1)
        self.act = nn.GELU()

    def _radial_grid(self, H, W, device, dtype):
        if dtype.is_complex:
            dtype = torch.float32
        fy = torch.linspace(-0.5, 0.5, steps=H, device=device, dtype=dtype)
        fx = torch.linspace(0.0, 0.5, steps=W // 2 + 1, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(fy, fx, indexing='ij')
        r = torch.sqrt(gy ** 2 + gx ** 2)
        r = torch.clamp(r / (r.max() + 1e-08), 0.0, 1.0)
        return r

    def _band_masks(self, H, W, device, dtype):
        r = self._radial_grid(H, W, device, dtype)
        low_gate = torch.sigmoid(10.0 * (0.2 - r))
        high_gate = torch.sigmoid(10.0 * (r - 0.45))
        mid_gate = torch.clamp(1.0 - low_gate - high_gate, 0.0, 1.0)
        return (low_gate[None, None, ...], mid_gate[None, None, ...], high_gate[None, None, ...])

    @torch.no_grad()
    def _freq_enhance(self, x):
        B, C, H, W = x.shape
        Hs, Ws = (max(1, H // self.down), max(1, W // self.down))
        x_ds = F.adaptive_avg_pool2d(x, (Hs, Ws))
        Xf = torch.fft.rfft2(x_ds, norm='ortho')
        low_m, mid_m, high_m = self._band_masks(Hs, Ws, x_ds.device, x_ds.dtype)
        low_m = low_m.repeat(1, C, 1, 1)
        mid_m = mid_m.repeat(1, C, 1, 1)
        high_m = high_m.repeat(1, C, 1, 1)
        power = Xf.real ** 2 + Xf.imag ** 2

        def band_energy(mask):
            band_power_sum = (power * mask).sum(dim=(-2, -1))
            mask_pixel_num = mask.sum(dim=(-2, -1)) + 1e-08
            return band_power_sum / mask_pixel_num
        E_low, E_mid, E_high = (band_energy(low_m), band_energy(mid_m), band_energy(high_m))
        std = torch.std(x_ds, dim=[2, 3], keepdim=True)
        s_low = E_low / (E_low + 1e-08)
        s_mid = E_mid / (E_mid + 1e-08)
        s_high = E_high / (E_high + 1e-08)
        gain_low = (1 - s_low).unsqueeze(-1).unsqueeze(-1) * (std + 1e-06)
        gain_mid = (1 - s_mid).unsqueeze(-1).unsqueeze(-1) * (std + 1e-06)
        gain_high = (1 - s_high).unsqueeze(-1).unsqueeze(-1) * (std + 1e-06)
        gain_map = gain_low * low_m + gain_mid * mid_m + gain_high * high_m
        Xf_enh = Xf * gain_map
        x_freq = torch.fft.irfft2(Xf_enh, s=(Hs, Ws), norm='ortho')
        x_freq = F.interpolate(x_freq, size=(H, W), mode='bilinear', align_corners=False)

        def rms(t):
            return t.flatten(2).pow(2).mean(dim=-1, keepdim=False).sqrt() + 1e-06
        scale = rms(x) / rms(x_freq)
        scale = scale.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
        x_freq = x_freq * scale
        return x_freq

    def forward(self, x):
        x_spa = self.dwconv(x)
        x_freq = self._freq_enhance(x)
        x = x_spa + x_freq
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return x

class FMFFM(nn.Module):

    def __init__(self, dim: int, growth_rate: float=2.0, beta: float=0.5):
        super().__init__()
        hidden = int(dim * growth_rate)
        self.conv1 = nn.Conv2d(dim, hidden, 1)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.conv2 = nn.Conv2d(hidden, dim, 1)
        self.act = nn.GELU()
        self.beta = beta
        self.eps = 1e-06

    def _energy_gate(self, x):
        std = x.flatten(2).std(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(self.eps)
        mean_std = std.mean(dim=1, keepdim=True)
        gate = (std / mean_std).detach().pow(self.beta)
        return x * gate

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.dwconv(y)
        y = self._energy_gate(y)
        y = self.act(y)
        y = self.conv2(y)
        return y

    @torch.no_grad()
    def reparam(self):
        return self

class PFF(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.pfa = PFA(dim, n_iter=3, init_grid_scale=8, var_scale_init=0.1, beta_init=10.0)
        self.pgca = PGCA(dim, qk_dim=24, num_heads=1, tau_base=1.0, var_alpha=1.0)
        self.res_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        residual = x
        affinity, prototype_feat = self.pfa(x)
        x = self.pgca(x, affinity, prototype_feat)
        x = F.gelu(self.res_conv(x))
        return x + residual

class DualBranchFusion(nn.Module):

    def __init__(self, dim=36):
        super().__init__()
        self.dim = dim
        self.split_conv = nn.Conv2d(dim, 2 * dim, 1)
        self.pff = PFF(dim)
        self.sfcm = SFCM(dim, down=2)
        self.fuse_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        x_split = self.split_conv(x_norm)
        x_pff, x_sfcm = torch.split(x_split, self.dim, dim=1)
        x_pff = self.pff(x_pff)
        x_sfcm = self.sfcm(x_sfcm)
        return self.fuse_conv(x_pff + x_sfcm)

class MFCB(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.branch_fusion = DualBranchFusion(dim)
        self.fmffm = FMFFM(dim)

    def forward(self, x):
        x = self.branch_fusion(x) + x
        x = self.fmffm(x) + x
        return x

@ARCH_REGISTRY.register()
class PSFNet(nn.Module):

    def __init__(self, dim=36, n_blocks=8, upscaling_factor=4):
        super().__init__()
        self.scale = upscaling_factor
        self.shallow_conv = nn.Conv2d(3, dim, 3, 1, 1)
        self.blocks = nn.Sequential(*[MFCB(dim) for _ in range(n_blocks)])
        self.reconstruct = nn.Sequential(nn.Conv2d(dim, 3 * upscaling_factor ** 2, 3, 1, 1), nn.PixelShuffle(upscaling_factor))

    def forward(self, x):
        x = self.shallow_conv(x)
        x = self.blocks(x) + x
        x = self.reconstruct(x)
        return x

    @torch.no_grad()
    def reparam(self):
        for m in self.modules():
            if isinstance(m, SFCM):
                pass
            if isinstance(m, FMFFM):
                m.reparam()

@torch.no_grad()
def reparam_all(model: nn.Module):
    for m in model.modules():
        if isinstance(m, FMFFM):
            m.reparam()
    return model
__all__ = ['PFA', 'PGCA', 'SFCM', 'PFF', 'FMFFM', 'DualBranchFusion', 'MFCB', 'PSFNet', 'reparam_all']
