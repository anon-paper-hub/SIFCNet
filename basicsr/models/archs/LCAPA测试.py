import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import os

# === Low模块 ===
class Low(nn.Module):
    def __init__(self):
        super(Low, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=2, stride=2, bias=False)
        self.conv.weight.data.fill_(0.25)

    def forward(self, x):
        return self.conv(x)

# === 通道注意力（LCA） ===
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        reduction = max(channel // 8, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

# === 像素注意力（LPA） ===
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        reduction = max(channel // 8, 1)
        self.pa = nn.Sequential(
            nn.Conv2d(channel, reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

# === 增强版Lp提取器 ===
class EnhancedLpExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super(EnhancedLpExtractor, self).__init__()
        self.low = Low()
        self.ca = CALayer(in_channels)
        self.pa = PALayer(in_channels)
        self.conv_out = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        low = self.low(x)  # [B,3,H/2,W/2]
        low_up = F.interpolate(low, size=x.shape[2:], mode='bilinear', align_corners=False)
        low_ca = self.ca(low_up)
        low_pa = self.pa(low_ca)
        lp = self.conv_out(low_pa)  # [B,1,H,W]
        return lp, low, low_up, low_ca, low_pa

# === 光照估计器 ===
class Illumination_Estimator(nn.Module):
    def __init__(self, n_fea_middle=3, n_fea_in=4, n_fea_out=3):
        super(Illumination_Estimator, self).__init__()
        self.lp_extractor = EnhancedLpExtractor()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(n_fea_middle, n_fea_middle, kernel_size=5, padding=2,
                                    bias=True, groups=n_fea_middle)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # enhanced Lp + 低频模块各中间量
        lp, low, low_up, low_ca, low_pa = self.lp_extractor(img)
        input_cat = torch.cat([img, lp], dim=1)  # [B,4,H,W]

        x_1 = self.conv1(input_cat)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return {
            'original': img,
            'low': low,
            'low_up': low_up,
            'low_ca': low_ca,
            'low_pa': low_pa,
            'lp': lp,
            'input_cat': input_cat,
            'illu_fea': illu_fea,
            'illu_map': illu_map
        }

# === 归一化辅助函数 ===
def normalize_tensor(t):
    t = t.detach().cpu()
    t = t - t.min()
    t = t / (t.max() + 1e-8)
    return t

# === 可视化辅助函数 ===
def show_tensor(tensor, title, cmap='gray'):
    t = tensor.squeeze()
    if t.ndim == 3 and t.shape[0] == 3:
        t = t.permute(1, 2, 0)
        plt.imshow(t)
    else:
        t = normalize_tensor(t)
        plt.imshow(t, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# === 主测试流程 ===
def main():
    # 读取图像
    img_path = '2.png'  # 替换为你的图片路径
    img = Image.open(img_path).convert('RGB')
    transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)  # [1,3,H,W]

    model = Illumination_Estimator()
    model.eval()
    with torch.no_grad():
        outs = model(img_tensor)

    # 逐个显示
    show_tensor(outs['original'], '1Original Image')
    show_tensor(outs['low'], '2Low-frequency (Low)')
    show_tensor(outs['low_up'], '3Upsampled Low-frequency (Low_up)')
    show_tensor(outs['low_ca'], '4After Channel Attention (Low_ca)')
    show_tensor(outs['low_pa'], '5After Pixel Attention (Low_pa)')
    show_tensor(outs['lp'], '6Enhanced Lp')
    #show_tensor(outs['input_cat'][:, :3], 'Input concat: RGB part')
    #show_tensor(outs['input_cat'][:, 3:], 'Input concat: Enhanced Lp part')
    show_tensor(outs['illu_fea'].mean(1, keepdim=True), '7Illumination Feature (illu_fea)')
    show_tensor(outs['illu_map'], '8Illumination Map (illu_map)')

if __name__ == '__main__':
    main()
