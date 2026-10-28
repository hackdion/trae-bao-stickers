#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seed 自由创作版 · 01「收到！」分层动画渲染器
- 纯 Pillow 超采样分层渲染（内部 1536，输出 512），角色各部件独立分层：
  头（含屏幕/表情）、躯干、双腿、后臂、敬礼手臂（大臂/小臂/手套两段关节）、
  对勾、信号波纹、星光、文字气泡均为独立图层/独立时间轴。
- 输出 48 帧 @约 24fps(42ms) 的无缝循环，并导出 PNG 静帧 / GIF / 全彩 WebP。
身份基准：官方 TRAE 宝-原图（黑色方盒屏幕头、青色发光菱形眼、黑色几何机身）
风格参考：Seed 本轮 GenerateImage 探索图（哑光软胶 3D、敬礼、浅薄荷背景）
"""
import math, os, sys
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops

# ---------------- 路径 ----------------
HERE = os.path.dirname(os.path.abspath(__file__))
KIND = os.path.dirname(HERE)                       # 01_收到
OUT  = os.path.join(KIND, "输出")
SRC  = os.path.join(KIND, "源文件")
FRD  = os.path.join(SRC, "frames")
REC  = os.path.join(KIND, "过程记录")
for d in (OUT, SRC, FRD, REC):
    os.makedirs(d, exist_ok=True)

# ---------------- 常量 ----------------
N   = 48                # 帧数
SS  = 3                 # 超采样倍率
W   = H = 512
BIG = W * SS            # 1536
DUR = 40                # 每帧毫秒（GIF 精度 10ms）→ 48 帧 1.92s
PEAK = 20               # 静帧取帧
FONT_CN = "/System/Library/Fonts/Hiragino Sans GB.ttc"   # index 2 = W6 粗体
FONT_EN = "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf"

# ---------------- 调色板 ----------------
BG0   = (248, 253, 252, 255)   # 背景顶
BG1   = (213, 238, 233, 255)   # 背景底
MINT  = (103, 235, 217, 255)
MINT2 = (52, 214, 195, 255)
MINT_D= (13, 64, 58, 255)
GOLD  = (255, 190, 74, 255)
D0    = (47, 51, 56, 255)      # 机身亮
D1    = (30, 33, 37, 255)      # 机身中
D2    = (17, 19, 22, 255)      # 机身暗
D3    = (9, 11, 13, 255)
SC0   = (17, 30, 31, 255)
SC1   = (4, 8, 9, 255)

def s(v):
    return int(round(v * SS))

# ---------------- 基础绘图工具 ----------------
def rr_mask(w, h, r):
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), radius=r, fill=255)
    return m

def grad_rr(w, h, r, c0, c1, vertical=True):
    """竖向线性渐变圆角矩形（c0 在左/上）。w,h 已为超采样像素。"""
    g = Image.linear_gradient("L")
    g = g.resize((w, h)) if vertical else g.rotate(90, expand=True).resize((w, h))
    top = Image.new("RGBA", (w, h), c0)
    bot = Image.new("RGBA", (w, h), c1)
    img = Image.composite(top, bot, g)
    if r:
        img.putalpha(ImageChops.multiply(img.getchannel("A"), rr_mask(w, h, r)))
    return img

def radial(w, h, c_in, c_out, r_in=0.0):
    g = Image.radial_gradient("L").resize((w, h))
    if r_in:
        g = g.point(lambda v: 0 if v < (1 - r_in) * 255 else min(255, int((v - (1 - r_in) * 255) / max(1, r_in * 255) * 255)))
    a = Image.new("RGBA", (w, h), c_in)
    b = Image.new("RGBA", (w, h), c_out)
    return Image.composite(a, b, g)

def blur(img, r):
    return img.filter(ImageFilter.GaussianBlur(r))

def paste(base, layer, cx, cy):
    """以 layer 中心对齐 (cx,cy)（最终像素坐标）粘贴。"""
    base.alpha_composite(layer, (s(cx) - layer.width // 2, s(cy) - layer.height // 2))

def rotated_part(sprite, deg):
    return sprite.rotate(deg, resample=Image.BICUBIC, expand=False)

# ---------------- 关节胶囊 ----------------
def capsule_sprite(length, width, c_hi, c_lo, shine=True, rim=False):
    """轴心(关节起点)在画布中心，肢体自中心向下；返回正方形 RGBA。"""
    c = int(length / 2 + width)
    cnv = Image.new("RGBA", (c * 2, c * 2), (0, 0, 0, 0))
    cx = c
    box = (cx - width // 2, c, cx + width // 2, c + length)
    limb = grad_rr(width, length, width // 2, c_hi, c_lo, vertical=False)
    cnv.alpha_composite(limb, (box[0], box[1]))
    if shine:
        sh = grad_rr(max(2, width // 5), int(length * 0.78), max(1, width // 10),
                     (255, 255, 255, 46), (255, 255, 255, 0), vertical=True)
        cnv.alpha_composite(sh, (box[0] + width // 4, box[1] + width // 5))
    if rim:
        rm = grad_rr(max(2, width // 7), int(length * 0.9), max(1, width // 14),
                     (103, 235, 217, 80), (103, 235, 217, 0), vertical=True)
        cnv.alpha_composite(rm, (box[2] - width // 4 - rm.width // 2, box[1] + width // 12))
    return cnv

def limb(base, A, B, width, c_hi, c_lo, rim=False, shine=True):
    L = int(round(math.hypot(B[0] - A[0], B[1] - A[1])))
    ang = math.degrees(math.atan2(B[1] - A[1], B[0] - A[0]))
    sp = capsule_sprite(s(L), s(width), c_hi, c_lo, shine=shine, rim=rim)
    base.alpha_composite(rotated_part(sp, 90 - ang),
                         (s(A[0]) - sp.width // 2, s(A[1]) - sp.height // 2))

def cuff(base, P, ang, w=30):
    """腕部同色含蓄环带。"""
    L = 8.0
    Q = (P[0] + L * math.sin(math.radians(ang)), P[1] + L * math.cos(math.radians(ang)))
    limb(base, P, Q, w, (48, 52, 57, 255), (29, 32, 36, 255), shine=True)

def mitten_sprite(size=40):
    """圆手套（自然下垂用）：腕点在画布中心，整体为光滑圆角手套，无外凸拇指。"""
    c = s(size * 1.3)
    img = Image.new("RGBA", (c * 2, c * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy = c, c
    pw, ph = s(size * 0.96), s(size * 1.0)
    d.rounded_rectangle((cx - pw // 2, cy, cx + pw // 2, cy + ph),
                        radius=s(size * 0.46), fill=(31, 34, 38, 255))
    # 内侧含蓄拇指（含在轮廓内，靠明暗暗示）
    d.ellipse((cx - pw // 2 + s(1), cy + s(size * 0.34),
               cx - pw // 2 + s(size * 0.34), cy + s(size * 0.74)), fill=(38, 42, 46, 255))
    return img

def salute_hand_sprite(size=46):
    """敬礼手掌：腕点在画布中心，并拢四指向下，拇指在局部 +x 侧（抬起后朝向头部）。"""
    c = s(size * 1.7)
    img = Image.new("RGBA", (c * 2, c * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy = c, c
    w = s(32)
    # 手指段（修长）
    d.rounded_rectangle((cx - w // 2, cy - s(14), cx + w // 2, cy + s(26)),
                        radius=s(13), fill=(32, 35, 39, 255))
    # 掌段（略宽）
    d.rounded_rectangle((cx - w // 2 - s(2), cy + s(18), cx + w // 2 + s(2), cy + s(46)),
                        radius=s(12), fill=(28, 31, 35, 255))
    # 拇指（局部 +x，斜向指尖）
    d.ellipse((cx + w // 2 - s(4), cy + s(16), cx + w // 2 + s(15), cy + s(40)),
              fill=(34, 38, 42, 255))
    # 两条指缝
    for gx in (-s(6), s(6)):
        d.line((cx + gx, cy - s(6), cx + gx, cy + s(22)), fill=(12, 14, 16, 120), width=s(2))
    # 指背受光
    hl = Image.new("RGBA", img.size, (0, 0, 0, 0))
    hd = ImageDraw.Draw(hl)
    hd.rounded_rectangle((cx - w // 3, cy - s(12), cx - w // 9, cy + s(20)),
                         radius=s(7), fill=(255, 255, 255, 30))
    img.alpha_composite(hl)
    return img

def place_sprite_alpha(base, sp, P, alpha):
    if alpha >= 0.995:
        r = sp
    else:
        r = sp.copy()
        r.putalpha(r.getchannel("A").point(lambda v: int(v * alpha)))
    base.alpha_composite(r, (s(P[0]) - r.width // 2, s(P[1]) - r.height // 2))

def place_mitten(base, P, ang, size=40, twist=0, open_palm=False, alpha=1.0):
    sp = salute_hand_sprite(size) if open_palm else mitten_sprite(size)
    place_sprite_alpha(base, rotated_part(sp, 90 - ang + twist), P, alpha)

# ---------------- 静态部件预渲染（最终像素坐标） ----------------
def build_bg():
    bg = grad_rr(BIG, BIG, 0, BG0, BG1, vertical=True)
    # 头后薄荷光晕
    glow = radial(s(560), s(560), (168, 242, 228, 150), (168, 242, 228, 0))
    bg.alpha_composite(glow, (s(256) - glow.width // 2, s(205) - glow.height // 2))
    # 柔光斑
    for cx, cy, r, a in ((92, 176, 86, 26), (428, 382, 104, 30), (420, 120, 52, 22)):
        b = radial(s(r * 2), s(r * 2), (255, 255, 255, a), (255, 255, 255, 0))
        bg.alpha_composite(b, (s(cx) - b.width // 2, s(cy) - b.height // 2))
    return bg

def build_legs():
    img = Image.new("RGBA", (BIG, BIG), (0, 0, 0, 0))
    for x in (220, 292):
        limb(img, (x, 346), (x, 442), 42, D0, D2)
        foot = grad_rr(s(60), s(44), s(22), D0, D2, vertical=False)
        img.alpha_composite(foot, (s(x) - foot.width // 2, s(448) - foot.height // 2))
        toe = Image.new("RGBA", (s(60), s(44)), (0, 0, 0, 0))
        td = ImageDraw.Draw(toe)
        td.rounded_rectangle((s(12), s(9), s(34), s(17)), radius=s(4), fill=(255, 255, 255, 26))
        toe.putalpha(ImageChops.multiply(toe.getchannel("A"), rr_mask(s(60), s(44), s(22))))
        img.alpha_composite(toe, (s(x) - s(30), s(448) - s(22)))
    return img

def build_torso():
    img = Image.new("RGBA", (BIG, BIG), (0, 0, 0, 0))
    body = grad_rr(s(164), s(138), s(46), D0, D2, vertical=True)
    img.alpha_composite(body, (s(256) - body.width // 2, s(252)))
    belly = grad_rr(s(122), s(86), s(30), (39, 43, 48, 255), (26, 29, 33, 255), vertical=True)
    img.alpha_composite(belly, (s(256) - belly.width // 2, s(286)))
    # 顶部颈窝阴影
    sh = radial(s(120), s(46), (0, 0, 0, 120), (0, 0, 0, 0))
    img.alpha_composite(sh, (s(256) - sh.width // 2, s(258) - s(6)))
    # 右侧青色轮廓光（收在身体边缘内）
    rim = blur(grad_rr(s(6), s(84), s(3), (103, 235, 217, 78), (103, 235, 217, 0)), s(2))
    img.alpha_composite(rim, (s(326), s(292)))
    # TRAE 铭牌
    badge = Image.new("RGBA", (s(72), s(34)), (0, 0, 0, 0))
    bd = ImageDraw.Draw(badge)
    bd.rounded_rectangle((0, 0, s(72) - 1, s(34) - 1), radius=s(13),
                         fill=(14, 20, 21, 255), outline=(88, 222, 206, 255), width=s(2))
    f = ImageFont.truetype(FONT_EN, s(17))
    bb = bd.textbbox((0, 0), "TRAE", font=f)
    bd.text((s(36) - (bb[2] - bb[0]) // 2 - bb[0], s(17) - (bb[3] - bb[1]) // 2 - bb[1] - s(1)),
            "TRAE", font=f, fill=(138, 244, 230, 255))
    img.alpha_composite(badge, (s(256) - s(36), s(330)))
    return img

def build_head():
    """头部硬件（不含屏幕表情）。中心 (256,195)。"""
    img = Image.new("RGBA", (BIG, BIG), (0, 0, 0, 0))
    # 两侧小耳块
    for x0, x1 in ((112, 132), (380, 400)):
        e = grad_rr(s(x1 - x0), s(64), s(9), D1, D2, vertical=True)
        img.alpha_composite(e, (s(x0), s(156)))
    # 外壳
    head = grad_rr(s(262), s(208), s(54), (49, 53, 58, 255), (18, 21, 24, 255), vertical=True)
    img.alpha_composite(head, (s(125), s(91)))
    # 顶部弧面高光
    top = grad_rr(s(230), s(78), s(34), (255, 255, 255, 30), (255, 255, 255, 0), vertical=True)
    top.putalpha(ImageChops.multiply(top.getchannel("A"), rr_mask(s(230), s(78), s(34))))
    img.alpha_composite(top, (s(141), s(101)))
    # 右侧青色边缘光
    rim = blur(grad_rr(s(10), s(150), s(5), (118, 240, 224, 110), (118, 240, 224, 0)), s(3))
    img.alpha_composite(rim, (s(372), s(120)))
    # 屏幕
    scr = grad_rr(s(214), s(156), s(34), SC0, SC1, vertical=True)
    # 屏幕内部径向聚光
    rg = radial(s(214), s(156), (14, 36, 36, 160), (14, 36, 36, 0))
    scr.alpha_composite(rg)
    # 屏幕内描边微光
    sd = ImageDraw.Draw(scr)
    sd.rounded_rectangle((s(2), s(2), s(211), s(153)), radius=s(32),
                         outline=(94, 226, 210, 70), width=s(2))
    # 玻璃斜反光
    glass = Image.new("RGBA", (s(214), s(156)), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glass)
    gd.polygon([(0, 0), (s(118), 0), (s(48), s(156)), (0, s(156))], fill=(255, 255, 255, 24))
    gd.polygon([(s(150), 0), (s(186), 0), (s(116), s(156)), (s(96), s(156))], fill=(255, 255, 255, 12))
    glass.putalpha(ImageChops.multiply(glass.getchannel("A"), rr_mask(s(214), s(156), s(34))))
    scr.alpha_composite(glass)
    img.alpha_composite(scr, (s(149), s(117)))
    # 底部状态灯（圆形遮罩）
    led = radial(s(18), s(18), (190, 255, 246, 255), (60, 220, 200, 0))
    lm = Image.new("L", (s(18), s(18)), 0)
    ImageDraw.Draw(lm).ellipse((0, 0, s(18) - 1, s(18) - 1), fill=255)
    led.putalpha(ImageChops.multiply(led.getchannel("A"), lm))
    img.alpha_composite(led, (s(344) - s(9), s(280) - s(9)))
    return img

# ---------------- 表情/特效精灵 ----------------
def diamond_sprite(h=46, glow=1.0):
    """发光菱形眼，中心即画面中心。"""
    w, hh = s(34), s(h)
    pad = s(34)
    cw = w + pad * 2
    img = Image.new("RGBA", (cw, hh + pad * 2), (0, 0, 0, 0))
    cx, cy = cw // 2, (hh + pad * 2) // 2
    halo = Image.new("RGBA", img.size, (0, 0, 0, 0))
    hd = ImageDraw.Draw(halo)
    hd.polygon([(cx, cy - hh // 2 - s(4)), (cx + w // 2 + s(6), cy),
                (cx, cy + hh // 2 + s(4)), (cx - w // 2 - s(6), cy)],
               fill=(90, 234, 218, int(150 * glow)))
    halo = blur(halo, s(7))
    img.alpha_composite(halo)
    d = ImageDraw.Draw(img)
    d.polygon([(cx, cy - hh // 2), (cx + w // 2, cy), (cx, cy + hh // 2), (cx - w // 2, cy)],
              fill=(232, 255, 251, 255))
    inner = Image.new("RGBA", img.size, (0, 0, 0, 0))
    idd = ImageDraw.Draw(inner)
    idd.polygon([(cx, cy - hh // 2), (cx + w // 2, cy), (cx, cy), (cx - w // 2, cy)],
                fill=(140, 245, 232, 255))
    img.alpha_composite(inner)
    return img

def check_sprite():
    """屏幕对勾，返回大图，中心约在 (256,236)。"""
    img = Image.new("RGBA", (s(120), s(96)), (0, 0, 0, 0))
    P = [(s(20), s(50)), (s(40), s(72)), (s(98), s(18))]
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).line(P, fill=(80, 235, 218, 200), width=s(17), joint="curve")
    glow = blur(glow, s(5))
    img.alpha_composite(glow)
    d = ImageDraw.Draw(img)
    d.line(P, fill=(120, 244, 229, 255), width=s(12), joint="curve")
    d.line(P, fill=(244, 255, 253, 255), width=s(5), joint="curve")
    return img

def check_mask(progress):
    m = Image.new("L", (s(120), s(96)), 0)
    d = ImageDraw.Draw(m)
    P = [(s(20), s(50)), (s(40), s(72)), (s(98), s(18))]
    seg = [math.hypot(P[1][0] - P[0][0], P[1][1] - P[0][1]),
           math.hypot(P[2][0] - P[1][0], P[2][1] - P[1][1])]
    tot = sum(seg)
    dist = max(0.0, min(1.0, progress)) * tot
    pts = [P[0]]
    for i in (0, 1):
        if dist <= 0:
            break
        if dist >= seg[i]:
            pts.append(P[i + 1]); dist -= seg[i]
        else:
            a = (P[i + 1][0] - P[i][0]) / seg[i]
            b = (P[i + 1][1] - P[i][1]) / seg[i]
            pts.append((P[i][0] + a * dist, P[i][1] + b * dist)); dist = 0
    if len(pts) >= 2:
        d.line(pts, fill=255, width=s(18), joint="curve")
        d.ellipse((pts[-1][0] - s(9), pts[-1][1] - s(9), pts[-1][0] + s(9), pts[-1][1] + s(9)), fill=255)
    return m

def star_sprite(color, size=40):
    c = s(size)
    img = Image.new("RGBA", (c * 2, c * 2), (0, 0, 0, 0))
    halo = Image.new("RGBA", img.size, (0, 0, 0, 0))
    hp = [(c, c - s(size) * 0.62), (c + s(size) * 0.16, c - s(size) * 0.16),
          (c + s(size) * 0.62, c), (c + s(size) * 0.16, c + s(size) * 0.16),
          (c, c + s(size) * 0.62), (c - s(size) * 0.16, c + s(size) * 0.16),
          (c - s(size) * 0.62, c), (c - s(size) * 0.16, c - s(size) * 0.16)]
    ImageDraw.Draw(halo).polygon(hp, fill=color[:3] + (160,))
    img.alpha_composite(blur(halo, s(4)))
    p = [(x, y) for x, y in hp]
    d = ImageDraw.Draw(img)
    d.polygon(p, fill=color)
    d.polygon([p[0], p[1], (c, c), p[7]], fill=(255, 255, 255, 255))
    return img

def arc_sprite(radius, width=6):
    ext = s(radius * 2 + 26)
    img = Image.new("RGBA", (ext, ext), (0, 0, 0, 0))
    q = s(13)
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).arc((q, q, ext - q, ext - q), -70, 70, fill=(90, 234, 218, 180), width=s(width + 3))
    img.alpha_composite(blur(glow, s(3)))
    d = ImageDraw.Draw(img)
    d.arc((q, q, ext - q, ext - q), -70, 70, fill=(225, 255, 250, 255), width=s(width))
    return img

# ---------------- 文字气泡 ----------------
def build_pill():
    txt = "收到！"
    f = ImageFont.truetype(FONT_CN, s(58), index=2)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    bb = probe.textbbox((0, 0), txt, font=f, stroke_width=s(3))
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pw, ph = tw + s(56), th + s(26)
    img = Image.new("RGBA", (pw + s(24), ph + s(24)), (0, 0, 0, 0))
    ox, oy = s(12), s(12)
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle((ox, oy + s(7), ox + pw, oy + ph + s(7)), radius=ph // 2,
                         fill=(9, 46, 42, 130))
    shadow = blur(shadow, s(4))
    img.alpha_composite(shadow)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((ox, oy, ox + pw, oy + ph), radius=ph // 2,
                        fill=(255, 255, 255, 255), outline=(14, 44, 40, 255), width=s(5))
    d.text((ox + pw // 2 - tw // 2 - bb[0], oy + ph // 2 - th // 2 - bb[1]),
           txt, font=f, fill=MINT_D, stroke_width=s(2), stroke_fill=(14, 80, 72, 255))
    st = star_sprite(GOLD, 22)
    img.alpha_composite(st, (ox + pw - s(6), oy - s(8)))
    return img

# ---------------- 缓动 ----------------
def ease_out_back(t):
    c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2

def ease_in_out(t):
    return 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2

def clamp01(v):
    return max(0.0, min(1.0, v))

# ---------------- 预渲染实例 ----------------
BG    = build_bg()
LEGS  = build_legs()
TORSO = build_torso()
HEAD0 = build_head()
PILL0 = build_pill()
DIAM  = diamond_sprite()
CHECK0= check_sprite()

def ground_shadow(scale_x):
    # 椭圆软投影：RGB 恒定，仅对 alpha 蒙版做大模糊，避免扁椭圆像“垫板”
    ew, eh = s(190 * scale_x), s(30)
    pad = s(30)
    w, h = ew + pad * 2, eh + pad * 2
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse((pad, pad, pad + ew - 1, pad + eh - 1), fill=64)
    mask = blur(mask, s(9))
    sh = Image.new("RGBA", (w, h), (38, 86, 78, 0))
    sh.putalpha(mask)
    return sh

def render_frame(f, downscale=True):
    cv = Image.new("RGBA", (BIG, BIG), (0, 0, 0, 0))
    cv.alpha_composite(BG)

    bob = math.sin(2 * math.pi * f / N) * 3.0          # 呼吸起伏（整周期无缝）
    # 地面投影
    gs = ground_shadow(1.0 - 0.05 * math.sin(2 * math.pi * f / N + math.pi / 2))
    cv.alpha_composite(gs, (s(256) - gs.width // 2, s(462) - gs.height // 2))

    # —— 后臂（画面左侧，自然小摆）——
    sw = math.sin(2 * math.pi * f / N) * 4.0
    limb(cv, (182, 276), (170 + sw, 332), 30, D0, D2)
    limb(cv, (170 + sw, 332), (178 + sw * 0.5, 380), 27, D0, D2)
    place_mitten(cv, (180 + sw * 0.5, 388), 96 + sw * 0.2, size=33)

    # —— 颈 / 腿 / 躯干（随呼吸，颈部在躯干之后被压住只露一截）——
    neck = grad_rr(s(42), s(34), s(8), D2, D3, vertical=True)
    cv.alpha_composite(neck, (s(256) - s(21), s(278) + s(bob)))
    cv.alpha_composite(LEGS, (0, s(bob)))
    cv.alpha_composite(TORSO, (0, s(bob)))

    # —— 敬礼手臂动作时间轴 ——
    if f < 6:
        r = 0.0
    elif f < 17:
        t = (f - 6) / 11
        r = 1 - (1 - t) ** 3                      # ease-out cubic，干净抬起不过冲
    elif f < 33:
        r = 1.0
    elif f < 43:
        r = 1.0 - ease_in_out((f - 33) / 10)
    else:
        r = 0.0
    r = clamp01(r)
    tap = 0.0
    if 15 <= f <= 21:                                   # 触碰到太阳穴的轻顿
        tap = math.sin((f - 15) / 6 * math.pi) * 3.0

    # 静息：手臂下垂；峰值：肘外展、腕到太阳穴(约368,184)
    a1 = 12 + (104 - 12) * r
    a2 = 2 + (214 - 2) * r - tap
    L1 = 82 - 8 * r
    L2 = 60 + 12 * r
    S = (336, 262 + bob * 0.4)
    E = (S[0] + L1 * math.sin(math.radians(a1)), S[1] + L1 * math.cos(math.radians(a1)))
    Wp = (E[0] + L2 * math.sin(math.radians(a2)), E[1] + L2 * math.cos(math.radians(a2)))
    # 手掌朝向：下垂 → 手指竖直朝上
    hb = clamp01((r - 0.25) / 0.5)
    hb = hb * hb * (3 - 2 * hb)
    hand_ang = a2 * (1 - hb) + 270 * hb

    # —— 头部（含屏幕表情，一起轻微点头）——
    head = Image.new("RGBA", (BIG, BIG), (0, 0, 0, 0))
    head.alpha_composite(HEAD0)
    # 眨眼（敬礼触顶瞬间）
    blink = 1.0
    if 17 <= f <= 19:
        blink = [0.15, 0.1, 0.55][f - 17]
    pulse = 1.0 + 0.14 * math.exp(-((f - 20) / 4.0) ** 2)
    for ex in (212, 300):
        eye = DIAM.resize((DIAM.width, max(2, int(DIAM.height * blink * pulse))), Image.BICUBIC)
        head.alpha_composite(eye, (s(ex) - eye.width // 2, s(180 + bob) - eye.height // 2 - s(bob)))
    # 屏幕对勾（描边出现 → 停留 → 消失）
    if 17 <= f < 36:
        if f < 22:
            prog = ease_out_back((f - 17) / 5)
            cm = check_mask(clamp01(prog))
            ck = CHECK0.copy(); ck.putalpha(ImageChops.multiply(ck.getchannel("A"), cm))
        elif f < 33:
            ck = CHECK0
        else:
            a = int(255 * (1 - (f - 33) / 3))
            ck = CHECK0.copy(); ck.putalpha(ImageChops.multiply(ck.getchannel("A"),
                                                                CHECK0.getchannel("A").point(lambda v: int(v * a / 255))))
        head.alpha_composite(ck, (s(256) - ck.width // 2, s(230) - ck.height // 2))
    # 头部整体微点
    tilt = -r * 2.2 - tap * 0.5
    head = head.rotate(tilt, resample=Image.BICUBIC, center=(s(256), s(195)))
    cv.alpha_composite(head, (0, s(bob)))

    # —— 敬礼手臂（在头部之后；手套在圆手套→张掌之间随动作淡变）——
    ao = clamp01((r - 0.30) / 0.40)
    ao = ao * ao * (3 - 2 * ao)
    limb(cv, S, E, 34, D0, D2, rim=True)
    limb(cv, E, Wp, 30, D0, D2, rim=True)
    if 1 - ao > 0.02:
        place_mitten(cv, Wp, a2, size=36, alpha=1 - ao)
    cuff(cv, Wp, a2, w=30)
    if ao > 0.02:
        place_mitten(cv, Wp, hand_ang, size=44, open_palm=True, alpha=ao)

    # —— 信号波纹 ——
    if 16 <= f < 36:
        for k in range(3):
            t0 = 16 + k * 3
            if f >= t0:
                u = (f - t0) / 12.0
                if u <= 1:
                    alpha = int(255 * (1 - u) * (0.4 + 0.6 * clamp01(u * 4)))
                    rad = 22 + 30 * u
                    ar = arc_sprite(rad).copy()
                    ar.putalpha(ImageChops.multiply(ar.getchannel("A"),
                                                    ar.getchannel("A").point(lambda v: int(v * alpha / 255))))
                    cv.alpha_composite(ar, (s(420) - ar.width // 2, s(138) - ar.height // 2))

    # —— 星光（常驻闪烁 + 确认爆发）——
    stars = [(88, 118, 26, MINT, 0.0), (96, 276, 20, GOLD, 1.6),
             (436, 318, 24, MINT, 2.6), (78, 372, 18, GOLD, 3.4),
             (132, 96, 18, GOLD, 4.2)]
    for x, y, sz, col, ph in stars:
        a = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(2 * math.pi * f / N + ph))
        st = star_sprite(col, sz).copy()
        st.putalpha(ImageChops.multiply(st.getchannel("A"), st.getchannel("A").point(lambda v: int(v * a))))
        cv.alpha_composite(st, (s(x) - st.width // 2, s(y) - st.height // 2))
    if 16 <= f < 30:
        u = (f - 16) / 14
        for x, y, sz, col, ph in [(150, 132, 22, MINT2, 0), (402, 104, 24, GOLD, 1.1),
                                  (424, 232, 18, MINT, 2.2), (116, 224, 16, GOLD, 0.6)]:
            pop = math.sin(clamp01(u) * math.pi)
            sc = 0.5 + pop * 0.8
            st0 = star_sprite(col, sz)
            st = st0.resize((max(2, int(st0.width * sc)), max(2, int(st0.height * sc))), Image.BICUBIC)
            st.putalpha(ImageChops.multiply(st.getchannel("A"), st.getchannel("A").point(lambda v: int(v * pop))))
            cv.alpha_composite(st, (s(x) - st.width // 2, s(y) - st.height // 2))

    # —— 文字气泡（弹入 → 停留 → 缩小消失，首尾无气泡保证无缝）——
    sc = 0.0
    if 8 <= f < 14:
        sc = ease_out_back((f - 8) / 6)
    elif 14 <= f < 42:
        sc = 1.0 + 0.025 * math.sin(2 * math.pi * (f - 14) / 28)
    elif 42 <= f < 48:
        sc = 1.0 - ease_in_out((f - 42) / 6)
    if sc > 0.12:
        pill = PILL0.resize((max(2, int(PILL0.width * sc)), max(2, int(PILL0.height * sc))), Image.BICUBIC)
        cv.alpha_composite(pill, (s(256) - pill.width // 2, s(56) - pill.height // 2))

    return cv.resize((W, H), Image.LANCZOS) if downscale else cv

# ---------------- 渲染 ----------------
def main():
    frames = []
    for f in range(N):
        im = render_frame(f).convert("RGB")
        frames.append(im)
        im.save(os.path.join(FRD, f"f_{f:03d}.png"))
        if f == PEAK:
            im.save(os.path.join(OUT, "01_收到_静帧.png"))
    # 超采样静帧源（真 1536，不降采样）
    big = render_frame(PEAK, downscale=False)
    big.save(os.path.join(SRC, "01_收到_静帧_超采样1536.png"))

    # 接触印样
    sheet = Image.new("RGB", (192 * 8, 192 * 6), (245, 250, 249))
    d = ImageDraw.Draw(sheet)
    for i, im in enumerate(frames):
        t = im.resize((184, 184), Image.LANCZOS)
        x, y = (i % 8) * 192 + 4, (i // 8) * 192 + 4
        sheet.paste(t, (x, y)); d.text((x + 4, y + 2), str(i), fill=(200, 40, 40))
    sheet.save(os.path.join(REC, "帧接触印样_v5终版.jpg"), quality=82)

    # 预览页用 96px 缩略图
    TH = os.path.join(SRC, "thumbs")
    os.makedirs(TH, exist_ok=True)
    for i, im in enumerate(frames):
        im.resize((96, 96), Image.LANCZOS).save(os.path.join(TH, f"t_{i:03d}.jpg"), quality=78, optimize=True)

    # GIF：逐帧本地 128 色调色板 + Floyd 抖动（体积/清晰度平衡，约 1.5MB）
    gifs = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT,
                        dither=Image.Dither.FLOYDSTEINBERG) for im in frames]
    gifs[0].save(os.path.join(OUT, "01_收到_动态.gif"), save_all=True, append_images=gifs[1:],
                 duration=DUR, loop=0, optimize=True, disposal=1)

    # 全彩 WebP
    frames[0].save(os.path.join(OUT, "01_收到_动态.webp"), save_all=True, append_images=frames[1:],
                   duration=DUR, loop=0, quality=82, method=6, allow_mixed=True, minimize_size=True)
    print("done:", N, "frames")

if __name__ == "__main__":
    main()
