import os
import glob
import math
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

# 如果你的类在其它文件，请改成：
# from your_model_file import RetinexFormer, IG_MSA

# --------- 工具函数 ----------
from basicsr.models.archs.RetinexFormer_arch import RetinexFormer


def _to_tensor(img_pil):
    """PIL.Image -> torch.FloatTensor [1,C,H,W], 0~1"""
    arr = np.array(img_pil.convert('RGB'), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
    return t

def _robust_norm_2d(x2d, qmin=0.01, qmax=0.99):
    """
    x2d: torch.Tensor [H,W], 进行分位数归一化到 [0,1] 再转 0~255
    更稳，不容易被极端值压扁对比度
    """
    x = x2d.detach().float()
    lo = torch.quantile(x, qmin)
    hi = torch.quantile(x, qmax)
    if (hi - lo) < 1e-6:
        x = x - x.min()
        if x.max() > 0: x = x / x.max()
        return (x * 255.0).clamp(0,255).byte().cpu().numpy()
    x = (x - lo) / (hi - lo)
    x = x.clamp(0,1)
    return (x * 255.0).byte().cpu().numpy()

def _save_gray(arr_uint8, save_path, colormap=False):
    """
    arr_uint8: numpy 2D uint8
    """
    if not colormap:
        Image.fromarray(arr_uint8, mode='L').save(save_path)
    else:
        # 伪彩：使用简单JET风格（无需额外依赖）
        import matplotlib
        import matplotlib.cm as cm
        cmap = cm.get_cmap('viridis')  # 论文里常见
        colored = (cmap(arr_uint8 / 255.0)[:, :, :3] * 255.0).astype(np.uint8)
        Image.fromarray(colored, mode='RGB').save(save_path)

def _make_grid(ch_list, grid_cols=4, pad=2, bg=0):
    """
    把多张 2D uint8 图拼成网格，返回 uint8 的 HxW
    ch_list: list of np.uint8 2D
    """
    if len(ch_list) == 0:
        raise ValueError("Empty feature list for grid.")
    h, w = ch_list[0].shape
    cols = grid_cols
    rows = math.ceil(len(ch_list) / cols)
    H = rows * h + (rows - 1) * pad
    W = cols * w + (cols - 1) * pad
    grid = np.full((H, W), bg, dtype=np.uint8)
    for idx, im in enumerate(ch_list):
        r = idx // cols
        c = idx % cols
        y = r * (h + pad)
        x = c * (w + pad)
        grid[y:y+h, x:x+w] = im
    return grid

# --------- Hook 定义 ----------

class IGMSAFeatureDumper:
    """
    注册到每个 IG_MSA 上的 forward hook。
    会把 IG_MSA 的输出 out([B,H,W,C]) 做可视化保存。
    """
    def __init__(self, module_name, out_root, save_modes=('mean','l2','topk'),
                 topk=8, grid_cols=4, colormap=False, tag_getter=lambda: "image"):
        self.module_name = module_name.replace('.', '-')
        self.out_root = out_root
        self.save_modes = save_modes
        self.topk = int(topk)
        self.grid_cols = int(grid_cols)
        self.colormap = bool(colormap)
        # 外部在跑每张图片前会更新 tag_getter() 的返回值（文件名）
        self.tag_getter = tag_getter
        os.makedirs(self.out_root, exist_ok=True)

    @torch.no_grad()
    def __call__(self, module, inp, out):
        """
        out: [B,H,W,C] 张量（IG_MSA.forward 的返回）
        """
        if not isinstance(out, torch.Tensor):
            return
        b, h, w, c = out.shape
        if b != 1:
            # 为了图清晰，这里假定 B=1；若你需要支持批量，可在文件名里加 batch 索引
            pass

        feat = out[0].permute(2,0,1).contiguous()  # [C,H,W]

        # 计算通道能量（挑 topk 用）
        # energy = sum(x^2) / (H*W)
        energy = (feat ** 2).mean(dim=(1,2))  # [C]
        topk_idx = torch.topk(energy, k=min(self.topk, c), largest=True).indices.tolist()

        tag = self.tag_getter()
        base = os.path.join(self.out_root, f"{tag}__{self.module_name}")

        # 1) mean
        if 'mean' in self.save_modes:
            mean_map = feat.mean(dim=0)                   # [H,W]
            mean_u8  = _robust_norm_2d(mean_map)
            _save_gray(mean_u8, base + "__mean.png", colormap=self.colormap)

        # 2) l2
        if 'l2' in self.save_modes:
            l2_map = torch.sqrt((feat ** 2).mean(dim=0))  # [H,W]
            l2_u8  = _robust_norm_2d(l2_map)
            _save_gray(l2_u8, base + "__l2.png", colormap=self.colormap)

        # 3) topk grid
        if 'topk' in self.save_modes and len(topk_idx) > 0:
            tiles = []
            for ci in topk_idx:
                norm_u8 = _robust_norm_2d(feat[ci])
                tiles.append(norm_u8)
            grid = _make_grid(tiles, grid_cols=self.grid_cols, pad=4, bg=0)
            _save_gray(grid, base + f"__top{len(topk_idx)}.png", colormap=self.colormap)

def _find_igmsa_modules(model):
    """返回 (name, module) 列表"""
    from types import FunctionType
    igs = []
    for name, m in model.named_modules():
        # 直接按类名判断，避免跨文件导入时的类型问题
        if m.__class__.__name__ == 'IG_MSA':
            igs.append((name, m))
    return igs

# --------- 主函数（你要调用的） ----------

def dump_igmsa_features_for_folder(
    image_folder: str,
    out_root: str,
    ckpt_path: str = None,
    device: str = None,
    save_modes=('mean','l2','topk'),
    topk: int = 8,
    grid_cols: int = 4,
    colormap: bool = True,
    img_glob: str = "*.*",
    n_feat: int = 31,
    stage: int = 1,
    num_blocks=(1,1,1)
):
    """
    扫描文件夹图片，跑一次模型，导出每个 IG_MSA 的输出特征图。

    参数：
      image_folder : 输入图片目录
      out_root     : 输出根目录（会自动为每张图建立子目录）
      ckpt_path    : 训练权重路径（可选；不传则用随机初始化）
      device       : 'cuda' / 'cpu'（默认自动判断）
      save_modes   : ('mean','l2','topk') 的子集
      topk         : 'topk' 可视化挑选的通道数
      grid_cols    : 'topk' 网格列数
      colormap     : 是否用伪彩（True 更“好看”）
      img_glob     : 匹配扩展名
      n_feat, stage, num_blocks : 用于构建 RetinexFormer 的结构参数
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) 建模（你的定义需已在同一作用域或可导入）
    model = RetinexFormer(in_channels=3, out_channels=3,
                          n_feat=n_feat, stage=stage,
                          num_blocks=list(num_blocks)).to(device).eval()

    # 2) 加载权重（如果提供）
    if ckpt_path is not None and os.path.isfile(ckpt_path):
        sd = torch.load(ckpt_path, map_location='cpu')
        # 兼容 DataParallel / key 带 'module.'
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        new_sd = {}
        for k,v in sd.items():
            nk = k.replace('module.', '')  # 去掉前缀
            new_sd[nk] = v
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        if ckpt_path:
            print(f"[warn] ckpt not found: {ckpt_path} ，将使用随机权重。")

    # 3) 注册 hooks
    handles = []
    current_tag = {"name": "image"}  # 闭包里可变引用
    def _get_tag():
        return current_tag["name"]

    out_dir_all = os.path.abspath(out_root)
    os.makedirs(out_dir_all, exist_ok=True)

    for name, m in _find_igmsa_modules(model):
        dumper = IGMSAFeatureDumper(
            module_name=name,
            out_root=out_dir_all,   # 统一到一个目录，文件名包含图名+层名
            save_modes=save_modes,
            topk=topk,
            grid_cols=grid_cols,
            colormap=colormap,
            tag_getter=_get_tag
        )
        h = m.register_forward_hook(dumper)
        handles.append(h)

    # 4) 遍历图片，逐张推理 + 导出
    img_paths = []
    for ext in ["*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"]:
        img_paths.extend(glob.glob(os.path.join(image_folder, ext)))
    if img_glob != "*.*":
        img_paths.extend(glob.glob(os.path.join(image_folder, img_glob)))
    img_paths = sorted(list(set(img_paths)))

    if len(img_paths) == 0:
        print(f"[warn] No images found in {image_folder}")
        # 退出前清理 hook
        for h in handles: h.remove()
        return

    with torch.no_grad():
        for p in img_paths:
            try:
                img = Image.open(p)
            except Exception as e:
                print(f"[skip] cannot open {p}: {e}")
                continue

            tag = os.path.splitext(os.path.basename(p))[0]
            current_tag["name"] = tag

            x = _to_tensor(img).to(device)  # [1,3,H,W], 0~1
            _ = model(x)  # 前向一次，所有 IG_MSA 的输出都会被 hook 保存

            print(f"[ok] {tag} -> features dumped at {out_dir_all}")

    # 5) 清理 hook
    for h in handles:
        h.remove()

# --------- 使用示例 ----------
if __name__ == "__main__":
    # 类似你调用 NIQE 的方式，这里调用一次：
    image_folder = r"F:/研究生/研究室/RetinexTransformer增强/Retinexformer-master/results/LOLxg/493.png"
    out_root     = r"F:/研究生/研究室/RetinexTransformer增强/Retinexformer-master/results/IGMSA_feats"
    ckpt_path    = "experiments/RetinexFormer_LOL_v1/best_psnr_18.39_1000.pth"  # 有模型权重就填路径，例如 r"C:/path/to/retinexformer_best.pth"

    dump_igmsa_features_for_folder(
        image_folder=image_folder,
        out_root=out_root,
        ckpt_path=ckpt_path,
        device=None,                 # 自动选 GPU/CPU
        save_modes=('mean','l2','topk'),
        topk=8,
        grid_cols=4,
        colormap=True,               # 伪彩色更上镜
        n_feat=31, stage=1, num_blocks=(1,1,1)
    )
