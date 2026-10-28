"""
TRAE宝表情包重制版 · 2.5D 分层动画引擎

设计说明（对应 README 里的技术说明）：
1. 原画保持不变，只把需要运动的局部（拳、手、披风、纸片、屏幕眼睛、像素爱心）
   用多边形/亮度阈值提取成独立 RGBA 图层，其余像素作为底板。
2. 图层的位移来自仿射变换（绕关节枢轴旋转、缩放、平移），不是整图抖动。
3. 需要柔性形变的部位（披风）用"边界钉住"的 remap 位移场，保证边缘不出现裂缝。
4. 屏幕眼睛是屏幕上唯一的亮区，用亮度阈值 + 连通域筛选提取，可单独缩放/位移/调暗。
5. 特效在 2 倍超采样画布上绘制后缩小，并叠加一次泛光，避免锯齿。
6. GIF 使用整片统一调色板 + 固定蓝噪声抖动，噪声在时间轴上不变化，因此不会闪烁。

依赖：numpy、opencv-python、Pillow
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

S = 512                      # 输出边长（1:1）
EPS = 1e-6

# ---------------------------------------------------------------- 基础读写

def load_rgb(path) -> np.ndarray:
    """读入为 float32 RGB，取值 0..1。"""
    im = Image.open(path).convert("RGB")
    if im.size != (S, S):
        im = im.resize((S, S), Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255.0


def to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)


def save_rgb(arr: np.ndarray, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(arr)).save(path, optimize=True)


# ---------------------------------------------------------------- 遮罩

def poly_mask(points, feather: float = 5.0, inset: float = 0.0, size: int = S) -> np.ndarray:
    """多边形遮罩，feather 为高斯羽化 sigma（像素）。inset>0 时向内收缩。"""
    m = np.zeros((size, size), np.uint8)
    pts = np.array(points, np.float32)
    if inset > 0:
        poly = pts.astype(np.int32).reshape(-1, 1, 2)
        m2 = np.zeros((size, size), np.uint8)
        cv2.fillPoly(m2, [poly], 255)
        k = int(inset) * 2 + 1
        m2 = cv2.erode(m2, np.ones((k, k), np.uint8))
        m = m2
    else:
        cv2.fillPoly(m, [pts.astype(np.int32)], 255)
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    return m.astype(np.float32) / 255.0


def ellipse_mask(cx, cy, rx, ry, feather=8.0, size=S) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    cv2.ellipse(m, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 255, -1)
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    return m.astype(np.float32) / 255.0


def bbox_window(mask: np.ndarray) -> np.ndarray:
    """在遮罩包围盒内生成 0..1 的余弦窗，用于把位移场钉在边界上。"""
    ys, xs = np.where(mask > 0.02)
    if len(xs) == 0:
        return np.zeros_like(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    win = np.zeros_like(mask)
    h = max(2, y1 - y0)
    w = max(2, x1 - x0)
    yy = (np.arange(S) - y0) / h
    xx = (np.arange(S) - x0) / w
    wx = np.clip(np.sin(np.pi * np.clip(xx, 0, 1)), 0, 1)
    wy = np.clip(np.sin(np.pi * np.clip(yy, 0, 1)), 0, 1)
    win = wy[:, None] * wx[None, :]
    return win


# ---------------------------------------------------------------- 图层

def extract(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """按遮罩提取预乘 RGBA 图层（避免旋转时出现黑边）。"""
    a = mask[..., None]
    return np.concatenate([img * a, a], axis=2).astype(np.float32)


def inpaint_plate(img: np.ndarray, mask: np.ndarray, grow: int = 5, radius: int = 7) -> np.ndarray:
    """把遮罩区域用 inpaint 抹平，作为图层移走后的底板。"""
    m = (mask > 0.02).astype(np.uint8) * 255
    if grow > 0:
        m = cv2.dilate(m, np.ones((grow * 2 + 1, grow * 2 + 1), np.uint8))
    bgr = to_uint8(img)
    out = cv2.inpaint(bgr, m, radius, cv2.INPAINT_TELEA)
    out = cv2.GaussianBlur(out, (0, 0), 0.6)
    return out.astype(np.float32) / 255.0


def affine(pivot, angle=0.0, scale=1.0, translate=(0.0, 0.0), size=S) -> np.ndarray:
    M = cv2.getRotationMatrix2D((float(pivot[0]), float(pivot[1])), float(angle), float(scale))
    M[0, 2] += translate[0]
    M[1, 2] += translate[1]
    return M


def warp_layer(layer: np.ndarray, M: np.ndarray, size=S, flags=cv2.INTER_CUBIC) -> np.ndarray:
    """仿射重采样。像素风图层可传 cv2.INTER_NEAREST 以保持硬边。"""
    out = cv2.warpAffine(layer, M, (size, size), flags=flags,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    np.clip(out, 0, 1, out=out)
    return out


def warp_field(layer: np.ndarray, fx, fy) -> np.ndarray:
    """用逐像素位移场重采样（fx, fy 为 512x512 目标坐标偏移）。可作用于 RGB 或预乘 RGBA。"""
    xs, ys = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
    mapx = (xs + fx).astype(np.float32)
    mapy = (ys + fy).astype(np.float32)
    out = cv2.remap(layer, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    np.clip(out, 0, 1, out=out)
    return out


def moving_glare(frame: np.ndarray, quad, t, angle=-0.55, width=46.0, alpha=0.11,
                 feather=6.0) -> np.ndarray:
    """
    屏幕玻璃反光：一条柔光带沿斜向缓慢扫过内屏，只在屏幕多边形内生效。
    用于给画面补一层持续、低幅度、面积较大的运动，避免出现完全静止的帧。
    """
    q = np.zeros((S, S), np.uint8)
    cv2.fillPoly(q, [np.array(quad, np.int32)], 255)
    qm = cv2.GaussianBlur(q, (0, 0), feather).astype(np.float32) / 255.0
    ys, xs = np.mgrid[0:S, 0:S].astype(np.float32)
    proj = xs * math.cos(angle) + ys * math.sin(angle)
    inner = proj[q > 0]
    lo, hi = float(inner.min()), float(inner.max())
    # 两条错开半个周期的光带，保证任何时刻屏幕上都有反光在移动
    band = np.zeros_like(proj)
    for k in (0.0, 0.5):
        u = (float(t) + k) % 1.0
        pos = lo - width + (hi - lo + width * 2) * u
        band += np.exp(-((proj - pos) / width) ** 2)
    a = (np.clip(band, 0, 1) * qm * alpha)[..., None]
    return 1.0 - (1.0 - frame) * (1.0 - a)


def camera(frame: np.ndarray, scale=1.0, dx=0.0, dy=0.0) -> np.ndarray:
    """整体镜头：轻微推近/位移，边缘用镜像补齐，避免出现透明边。"""
    if abs(scale - 1) < 1e-5 and abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return frame
    cx = S / 2.0
    M = np.array([[scale, 0, cx - scale * cx + dx],
                  [0, scale, cx - scale * cx + dy]], np.float32)
    return cv2.warpAffine(frame, M, (S, S), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REFLECT_101)


def over(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    a = layer[..., 3:4]
    return layer[..., :3] + base * (1.0 - a)


def screen_blend(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """发光用的 screen 叠加。layer 为预乘 RGBA。"""
    a = layer[..., 3:4]
    rgb = np.clip(layer[..., :3] / np.maximum(a, EPS), 0, 1)
    return 1.0 - (1.0 - base) * (1.0 - rgb * a)


# ---------------------------------------------------------------- 屏幕眼睛

def _components(mask_u8, min_area=180, max_side=130):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    keep = np.zeros_like(mask_u8)
    kept = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        if w > max_side or h > max_side:
            continue
        if area / max(1.0, w * h) < 0.18:
            continue
        keep[lab == i] = 255
        kept.append((x, y, w, h, area))
    return keep, kept


class ScreenEyes:
    """从内屏亮区提取眼睛图层，可独立做缩放、位移、明暗、发光。"""

    def __init__(self, img, quad, thr=0.72, grow=3, blur=1.2):
        q = np.zeros((S, S), np.uint8)
        cv2.fillPoly(q, [np.array(quad, np.int32)], 255)
        luma = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2GRAY)
        bright = ((luma.astype(np.float32) / 255.0) > thr).astype(np.uint8) * 255
        bright = cv2.bitwise_and(bright, q)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        keep, kept = _components(bright)
        if keep.sum() == 0:
            raise RuntimeError("未能从屏幕提取眼睛亮区，请检查 quad 或阈值")
        if grow > 0:
            keep = cv2.dilate(keep, np.ones((grow * 2 + 1, grow * 2 + 1), np.uint8))
        keep = cv2.bitwise_and(keep, q)
        m = cv2.GaussianBlur(keep, (0, 0), blur).astype(np.float32) / 255.0
        self.mask = m
        self.layer = extract(img, m)
        self.plate = inpaint_plate(img, m, grow=max(3, grow), radius=5)
        self.bbox = [int(v) for v in cv2.boundingRect(keep)]
        ys, xs = np.where(keep > 0)
        self.center = (float(xs.mean()), float(ys.mean())) if len(xs) else (S / 2, S / 2)
        self.stats = kept

    def layer_at(self, scale=1.0, dx=0.0, dy=0.0, dim=1.0, blink=0.0,
                 squint=1.0, pivot=None, matrix=None):
        """返回变换后的眼睛图层（预乘 RGBA）。"""
        piv = pivot if pivot is not None else self.center
        if matrix is None:
            sy = scale * squint * (1.0 - 0.92 * blink)
            M = np.array([[scale, 0, piv[0] - scale * piv[0] + dx],
                          [0, sy, piv[1] - sy * piv[1] + dy]], np.float32)
        else:
            M = matrix
        w = warp_layer(self.layer, M)
        if dim != 1.0:
            w[..., :3] *= max(0.0, dim)
        return w

    def render(self, t, scale=1.0, dx=0.0, dy=0.0, dim=1.0, glow=0.0,
               blink=0.0, squint=1.0, pivot=None, base=None, matrix=None):
        """把眼睛合成到 (base 或 自带的空屏底板) 上。"""
        w = self.layer_at(scale, dx, dy, dim, blink, squint, pivot, matrix)
        out = over(self.plate if base is None else base, w)
        if glow > 0:
            g = self.layer.copy()
            g[..., :3] *= glow
            g = cv2.GaussianBlur(g, (0, 0), 7)
            out = np.clip(screen_blend(out, g * 0.55), 0, 1)
        return out


# ---------------------------------------------------------------- 特效画布

class FX:
    """2 倍超采样绘制，结束时可选择加泛光。"""

    def __init__(self, ss=2):
        self.ss = ss
        self.im = Image.new("RGBA", (S * ss, S * ss), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im, "RGBA")

    def _p(self, xy):
        return (xy[0] * self.ss, xy[1] * self.ss)

    def circle(self, x, y, r, color):
        self.d.ellipse((self._p((x - r, y - r))[0], self._p((x - r, y - r))[1],
                        self._p((x + r, y + r))[0], self._p((x + r, y + r))[1]), fill=color)

    def line(self, pts, color, width=2, joint="curve"):
        self.d.line([self._p(p) for p in pts], fill=color, width=max(1, int(width * self.ss)), joint=joint)

    def tapered(self, x0, y0, x1, y1, width, color, steps=8):
        """一端粗一端细的速度线。"""
        for i in range(steps):
            u0, u1 = i / steps, (i + 1) / steps
            w0 = width * (1 - u0) ** 0.8
            ax, ay = x0 + (x1 - x0) * u0, y0 + (y1 - y0) * u0
            bx, by = x0 + (x1 - x0) * u1, y0 + (y1 - y0) * u1
            dx, dy = (bx - ax), (by - ay)
            ln = math.hypot(dx, dy) + EPS
            nx, ny = -dy / ln, dx / ln
            pts = [(ax + nx * w0, ay + ny * w0), (ax - nx * w0, ay - ny * w0),
                   (bx - nx * w0 * 0.6, by - ny * w0 * 0.6), (bx + nx * w0 * 0.6, by + ny * w0 * 0.6)]
            self.d.polygon([self._p(p) for p in pts], fill=color)

    def heart(self, x, y, size, color, rot=0.0, outline=False, width=1.6):
        pts = []
        for deg in range(0, 360, 6):
            a = math.radians(deg)
            # 经典心形参数方程（图像坐标 y 向下，故取反得到尖朝下）
            hx = 16 * math.sin(a) ** 3
            hy = -(13 * math.cos(a) - 5 * math.cos(2 * a) - 2 * math.cos(3 * a) - math.cos(4 * a))
            ca, sa = math.cos(rot), math.sin(rot)
            px = x + (hx * ca - hy * sa) * size / 16.0
            py = y + (hx * sa + hy * ca) * size / 16.0
            pts.append(self._p((px, py)))
        if outline:
            pts.append(pts[0])
            self.d.line(pts, fill=color, width=max(1, int(width * self.ss)), joint="curve")
        else:
            self.d.polygon(pts, fill=color)

    def glow_blob(self, x, y, r, color):
        for i in range(4):
            rr = r * (1 + i * 0.55)
            a = int(color[3] * (0.5 ** (i + 1)))
            self.circle(x, y, rr, (color[0], color[1], color[2], a))

    def finish(self, bloom=0.55, bloom_sigma=6.0):
        im = self.im.resize((S, S), Image.LANCZOS)
        arr = np.asarray(im).astype(np.float32) / 255.0
        if bloom > 0:
            bl = im.filter(ImageFilter.GaussianBlur(bloom_sigma))
            b = np.asarray(bl).astype(np.float32) / 255.0
            b[..., :3] *= b[..., 3:4]
            arr[..., :3] += b[..., :3] * bloom
        arr[..., :3] = np.clip(arr[..., :3], 0, 1)
        # 统一返回预乘 RGBA，与 over() 的约定保持一致
        arr[..., :3] *= arr[..., 3:4]
        return arr


# ---------------------------------------------------------------- 文字

FONT_CANDIDATES = [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
]


def _font(size):
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def name_chip(word: str, margin=16, accent=(255, 132, 66)) -> np.ndarray:
    """统一的系列角标：深色玻璃片 + 橙色强调条 + 中文短词。"""
    ss = 2
    im = Image.new("RGBA", (S * ss, S * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(im, "RGBA")
    font = _font(21 * ss)
    tw = d.textbbox((0, 0), word, font=font)[2]
    pad_l, pad_r, h = 14, 16, 34
    w = int(tw / ss) + pad_l + pad_r + 6
    x1 = S - margin
    y1 = S - margin
    x0 = x1 - w
    y0 = y1 - h
    d.rounded_rectangle((x0 * ss, y0 * ss, x1 * ss, y1 * ss), radius=7 * ss,
                        fill=(10, 12, 18, 168), outline=(255, 255, 255, 46), width=ss)
    d.rounded_rectangle((x0 * ss + 5 * ss, y0 * ss + 9 * ss, x0 * ss + 8 * ss, y1 * ss - 9 * ss),
                        radius=2 * ss, fill=accent + (235,))
    d.text(((x0 + pad_l) * ss, (y0 + h / 2) * ss), word, font=font, anchor="lm",
           fill=(255, 255, 255, 246))
    im = im.resize((S, S), Image.LANCZOS)
    arr = np.asarray(im).astype(np.float32) / 255.0
    arr[..., :3] *= arr[..., 3:4]      # 预乘，避免软边处颜色发灰
    return arr


# ---------------------------------------------------------------- 导出

def _pil_frames(frames):
    return [Image.fromarray(to_uint8(f)) for f in frames]


def build_palette(frames, colors=256, step=3):
    """所有帧共用一张调色板，保证循环时颜色稳定。"""
    sample = Image.new("RGB", (S * 3, S * 3))
    n = min(9, len(frames))
    for i in range(n):
        idx = int(i * (len(frames) - 1) / max(1, n - 1))
        small = Image.fromarray(to_uint8(frames[idx])[::step, ::step])
        small = small.resize((S, S), Image.NEAREST)
        sample.paste(small, ((i % 3) * S, (i // 3) * S))
    pal = sample.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    return pal


def quantize_frames(frames, pal, noise_amp=0.010, seed=7):
    """
    统一调色板 + 固定蓝噪声抖动。
    噪声在时间轴上完全不变化，所以不会出现帧间闪烁；只把渐变色带打散成静态颗粒。
    """
    rng = np.random.default_rng(seed)
    # 蓝噪声近似：白噪声减去低频分量，保留高频颗粒
    white = rng.normal(0, 1, (S, S, 3)).astype(np.float32)
    low = cv2.GaussianBlur(white, (0, 0), 2.2)
    blue = (white - low * 0.75) * noise_amp
    out = []
    for f in frames:
        noisy = np.clip(f + blue, 0, 1)
        im = Image.fromarray(to_uint8(noisy))
        out.append(im.quantize(palette=pal, dither=Image.Dither.NONE))
    return out


def export_gif(frames, path, duration_ms, colors=256, noise_amp=0.010):
    pal = build_palette(frames, colors=colors - 1)
    pf = quantize_frames(frames, pal, noise_amp=noise_amp)
    # disposal=1 + optimize=True：每帧只重写变化区域，体积约为整帧重写时的 1/7；
    # 因为每帧都不同，Pillow 不会丢帧（构建后用 verify_outputs.py 复核帧数）。
    pf[0].save(path, save_all=True, append_images=pf[1:], duration=duration_ms,
               loop=0, disposal=1, optimize=True)
    return Path(path).stat().st_size


def export_webp(frames, path, duration_ms, quality=90):
    ims = _pil_frames(frames)
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=duration_ms,
                loop=0, quality=quality, method=6)
    return Path(path).stat().st_size


# ---------------------------------------------------------------- 缓动

def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def smoothstep(x):
    x = clamp01(np.asarray(x, dtype=np.float64))
    return x * x * (3 - 2 * x)


def beat(t, k=1.0, phase=0.0):
    """周期性节拍：0..1 正弦正半周，t=0 与 t=1 完全衔接。"""
    return np.clip(np.sin(2 * math.pi * k * t + phase), 0, 1)


def punch(t, k=1.0, phase=0.0, sharp=0.55):
    """快起慢落的打击曲线，用于出拳/心跳。"""
    return beat(t, k, phase) ** sharp


def ease_pulse(t, center, width):
    """在 t=center 附近出现一次的平滑脉冲（循环边界处为 0）。"""
    d = np.abs(((t - center + 0.5) % 1.0) - 0.5)
    return smoothstep(np.clip(1 - d / width, 0, 1))


def key_curve(t, keys):
    """
    关键帧曲线：keys=[(t, value), ...]，段间用 smoothstep 过渡，首尾自动环绕。
    用来编排"快速抬起—回收—停顿—回到起点"这类非正弦动作。
    """
    pts = sorted((float(a) % 1.0, float(b)) for a, b in keys)
    if len(pts) == 1:
        return pts[0][1]
    tt = float(t) % 1.0
    prev, nxt = None, None
    if tt < pts[0][0]:
        prev = (pts[-1][0] - 1.0, pts[-1][1])
        nxt = pts[0]
    else:
        for i, (a, b) in enumerate(zip(pts, pts[1:])):
            if a[0] <= tt <= b[0]:
                prev, nxt = (a[0], a[1]), (b[0], b[1])
                break
        if prev is None:
            prev = pts[-1]
            nxt = (pts[0][0] + 1.0, pts[0][1])
    span = nxt[0] - prev[0]
    if span <= 1e-9:
        return nxt[1]
    u = float(smoothstep((tt - prev[0]) / span))
    return prev[1] + (nxt[1] - prev[1]) * u
