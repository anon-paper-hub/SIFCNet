import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx

# import cv2
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")

def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]

class Low(nn.Module):
    def __init__(self):
        super(Low, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=2, stride=2, bias=False)
        self.conv.weight.data.fill_(0.25)  # Haar 平均

    def forward(self, x):
        return self.conv(x)

class ColorNet(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=32):
        super(ColorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  
            nn.Conv2d(hidden_dim, in_channels, 1),  
            nn.Tanh()  
        )

    def forward(self, x):
        shift = self.net(x)          
        x_corrected = x + shift      
        return x_corrected.clamp(0, 1)

class SemanticExtractor(nn.Module):
    def __init__(self, in_channels=3, n_feat=31):
        super(SemanticExtractor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, n_feat, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, target_size=None):
        feat = self.encoder(x)
        if target_size is not None:
            feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
        return feat

class LFIE(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super(LFIE, self).__init__()
        self.low_freq_extractor = Low()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        B, C, H, W = img.shape

        with torch.no_grad():
            low = self.low_freq_extractor(img)  # B,3,H/2,W/2
            low_upsampled = F.interpolate(low, size=(H, W), mode='bilinear', align_corners=False)
            mean_c = low_upsampled.mean(dim=1, keepdim=True)  # B,1,H,W

        input = torch.cat([img, mean_c], dim=1)
        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


class SIF(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or 4*dim
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )
    def forward(self, x):
        return self.net(x)

class SIFModule(nn.Module):
    def __init__(self, dim):
        super(SEModule, self).__init__()
        self.dim = dim
        self.Wk = nn.Conv2d(dim, dim, 1, bias=False)
        self.Wq = nn.Conv2d(dim, dim, 1, bias=False)
        self.Wv = nn.Conv2d(dim, dim, 1, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.ffn = SIF_FFN(dim)

    def forward(self, illu_fea, sem_fea):
        B, C, H, W = illu_fea.shape
        K = self.Wk(illu_fea).view(B, C, H*W)
        Q = self.Wq(sem_fea).view(B, C, H*W)
        V = self.Wv(illu_fea).view(B, C, H*W)

        K = self.norm(K.transpose(1,2)).transpose(1,2)
        Q = self.norm(Q.transpose(1,2)).transpose(1,2)

        Ab = torch.matmul(Q, K.transpose(1,2)) / math.sqrt(C)
        Ab = F.softmax(Ab, dim=-1)

        V_out = torch.matmul(Ab, V) + V
        V_out = V_out.view(B, C, H, W)
        out = self.ffn(V_out)
        return out  # [B, C, H, W]

class FG_MSA(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.dim = dim
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.sif = SIFModule(dim)

    def forward(self, x_in, illu_fea_trans, sem_fea_trans=None):
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h*w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        if sem_fea_trans is not None:
            prior = self.sif(illu_fea_trans.permute(0,3,1,2), sem_fea_trans.permute(0,3,1,2))
        else:
            prior = self.sif(illu_fea_trans.permute(0,3,1,2), illu_fea_trans.permute(0,3,1,2))
        guidance = prior.permute(0,2,3,1).reshape(b, h*w, c)

        q, k, v, guidance = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, guidance))
        v = v * guidance 

        q = q.transpose(-2,-1)
        k = k.transpose(-2,-1)
        v = v.transpose(-2,-1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2,-1))
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x.permute(0,3,1,2)
        x = x.reshape(b, h*w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0,3,1,2)).permute(0,2,3,1)
        out = out_c + out_p
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                    bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)

class MGAM(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
            illu_ch=None,   
            sem_ch=None    
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        if (sem_ch is not None) and (illu_ch is not None) and (sem_ch != illu_ch):
            self.sem_align = nn.Conv2d(sem_ch, illu_ch, kernel_size=1, bias=False)
        else:
            self.sem_align = None

        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                FG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea, sem_fea=None):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        sem_fea: [b,c,h,w] or None
        return out: [b,c,h,w]
        """
        if sem_fea is not None:
            if sem_fea.shape[-2:] != illu_fea.shape[-2:]:
                sem_fea = F.interpolate(
                    sem_fea, size=illu_fea.shape[-2:], mode='bilinear', align_corners=False
                )
            if self.sem_align is not None:
                sem_fea = self.sem_align(sem_fea)

        x_hw = x.permute(0, 2, 3, 1)
        illu_hw = illu_fea.permute(0, 2, 3, 1)
        sem_hw  = sem_fea.permute(0, 2, 3, 1) if sem_fea is not None else None

        for (attn, ff) in self.blocks:
            x_hw = attn(
                x_hw,
                illu_fea_trans=illu_hw,
                sem_fea_trans=sem_hw
            ) + x_hw
            x_hw = ff(x_hw) + x_hw

        out = x_hw.permute(0, 3, 1, 2)  # [b, c, h, w]
        return out

class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)
        self.sem_proj = nn.Conv2d(dim, dim, 1, 1, bias=False)
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                MGAM(
                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim, illu_ch=dim_level, sem_ch=self.dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        self.bottleneck = MGAM(
            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1], illu_ch=dim_level, sem_ch=self.dim)

        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                MGAM(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim, illu_ch=dim_level // 2, sem_ch=self.dim),
            ]))
            dim_level //= 2

        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea, sem_fea):
        """
        x:          [b,c,h,w]         
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """

        fea = self.embedding(x)
        if sem_fea.shape[-2:] != fea.shape[-2:]:
            sem_fea = F.interpolate(sem_fea, size=fea.shape[-2:], mode='bilinear', align_corners=False)
        fea = fea + self.sem_proj(sem_fea)

        fea_encoder = []
        illu_fea_list = []
        sem_fea_list = []  
        for (AttnBlock, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            if sem_fea.shape[-2:] != illu_fea.shape[-2:]:
                sem_fea = F.interpolate(sem_fea, size=illu_fea.shape[-2:], mode='bilinear', align_corners=False)
            fea = AttnBlock(fea, illu_fea, sem_fea)
            fea_encoder.append(fea)
            illu_fea_list.append(illu_fea)
            sem_fea_list.append(sem_fea)

            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)
            sem_fea = F.interpolate(sem_fea, size=illu_fea.shape[-2:], mode='bilinear', align_corners=False)
        fea = self.bottleneck(fea, illu_fea, sem_fea)
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level - 1 - i]
            sem_fea  = sem_fea_list[self.level - 1 - i]
            fea = LeWinBlcok(fea, illu_fea, sem_fea)  

        out = self.mapping(fea) + x
        return out

class SIFC_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        super(SIFC_Stage, self).__init__()
        self.color_corrector = ColorNet(in_channels)
        self.estimator = LFIE(n_feat)
        self.semantic_extractor = SemanticExtractor(in_channels, n_feat)  
        self.denoiser = Denoiser(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level,
                                 num_blocks=num_blocks)

    def forward(self, img):
        # img:        b,c=3,h,w  

        # illu_fea:   b,c,h,w    Flu
        # illu_map:   b,c=3,h,w  Lbar

        illu_fea, illu_map = self.estimator(img)
        input_img = img * illu_map + img
        corrected_img = self.color_corrector(input_img)
        B, C, H, W = img.shape
        sem_fea = self.semantic_extractor(input_img, target_size=(H, W))
        output_img = self.denoiser(corrected_img, illu_fea, sem_fea)
        return output_img

class SIFCNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[1, 1, 1]):
        super(SIFCNet, self).__init__()
        self.stage = stage
        modules_body = [
            SIFC_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2,
                                       num_blocks=num_blocks)
            for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        out = self.body(x)
        return out

