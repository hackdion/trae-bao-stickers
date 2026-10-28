"""
TRAE宝表情包重制版 · 主构建脚本

用法：
    python scripts/build_remaster.py            # 全部六张
    python scripts/build_remaster.py --only 01  # 只做一张
    python scripts/build_remaster.py --debug    # 额外输出分层/遮罩调试图

原画来源：../TRAE宝表情包v5/*_原画.png
    （用户此前用 ChatGPT 生成、并已作为既有成果保存的参考图，本次只读引用，不改动原文件）
本次重制：分层、遮罩、局部运动、特效编排、调色与导出均由本脚本完成。

分层策略（每张共用同一套规则）：
  · 允许"只放大"的图层（拳、手、爱心、披风、纸片）：底板保留原图，放大图层直接盖住原位置，不挖洞、不留接缝。
  · 必须位移的图层（屏幕眼睛）：屏幕底色是平滑暗面，用 inpaint 抹掉后重绘完全无痕。
  · 必须下沉的图层（04 头部）：只在贴桌面的底边做锚定压缩，露出的一条窄边由 inpaint 补齐。
  · 披风用"边界钉住"的位移场直接作用于整帧，边缘位移为 0，不会裂缝。
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trae_rig as R

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "TRAE宝表情包v5"
OUT_GIF = ROOT / "gif"
OUT_WEBP = ROOT / "webp"
OUT_PNG = ROOT / "png"
DEBUG = Path("/tmp/trae_build/debug")


@dataclass
class Spec:
    sid: str
    name: str
    chip: str
    frames: int
    duration: int
    hero: float          # 取第几成的时间点作为 PNG 预览帧


# GIF 的帧时长以 10ms 为单位存储，这里的 duration 全部取 10 的整数倍，
# 保证脚本标注的时长与播放器实际读到的一致。
SPECS = [
    Spec("01", "收到", "收到", 24, 50, 0.125),
    Spec("02", "思考", "思考", 30, 50, 0.550),
    Spec("03", "震惊", "震惊", 24, 40, 0.200),
    Spec("04", "累瘫", "累瘫", 32, 60, 0.350),
    Spec("05", "冲刺", "冲刺", 20, 40, 0.250),
    Spec("06", "辛苦啦", "辛苦啦", 30, 50, 0.200),
]


def load(name: str) -> np.ndarray:
    return R.load_rgb(SRC / f"{name}_原画.png")


def debug_overlay(img, items, path):
    DEBUG.mkdir(parents=True, exist_ok=True)
    arr = img.copy()
    for mask, color in items:
        tint = np.array(color[:3], np.float32) / 255.0
        a = mask[..., None] * 0.22
        arr = arr * (1 - a) + tint * a
    base = Image.fromarray(R.to_uint8(np.clip(arr, 0, 1)))
    d = ImageDraw.Draw(base, "RGBA")
    for mask, color in items:
        m = (mask > 0.35).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            pts = [tuple(map(int, p[0])) for p in c]
            if len(pts) > 2:
                d.line(pts + [pts[0]], fill=color + (255,), width=2)
    base.save(path)


# ================================================================ 01 收到
def build_01():
    sp = SPECS[0]
    img = load("01_收到")

    arm_poly = [(99, 250), (102, 210), (113, 180), (131, 162), (147, 141), (166, 139),
                (181, 151), (187, 171), (199, 186), (212, 206), (221, 232), (233, 262),
                (249, 300), (262, 344), (244, 365), (219, 352), (199, 330), (183, 307),
                (159, 291), (135, 285), (115, 271)]
    arm_mask = R.poly_mask(arm_poly, feather=2.2)
    arm = R.extract(img, arm_mask)

    eyes = R.ScreenEyes(img, quad=[(256, 199), (440, 239), (433, 331), (245, 301)], thr=0.70)
    base = eyes.plate

    piv = (258, 420)   # 支点取在手臂更下方：放大时拳头同时明显向上抬，而不只是变大
    lift_keys = [(0.00, 0.20), (0.060, 1.00), (0.145, 0.60), (0.220, 0.95),
                 (0.360, 0.38), (0.480, 0.20), (0.600, 0.36), (0.720, 0.17),
                 (0.860, 0.31), (1.00, 0.20)]

    def frame(i):
        t = i / sp.frames
        lift = R.key_curve(t, lift_keys) + 0.05 * math.sin(2 * math.pi * t)
        p = max(0.0, (lift - 0.35) / 0.65)
        sway = 0.5 - 0.5 * math.cos(2 * math.pi * t)

        out = eyes.render(t, scale=1.0 + 0.020 + 0.030 * lift, glow=0.10 + 0.30 * lift)
        out = R.over(out, R.warp_layer(arm, R.affine(piv, 0.0, 1.0 + 0.066 * lift)))

        fx = R.FX()
        cx, cy = 138, 205
        for k in range(6):                       # 出拳速度线
            a = -2.55 + k * 0.30
            r0 = 74 + 18 * lift
            r1 = r0 + 18 + 26 * p
            fx.tapered(cx + r0 * math.cos(a), cy + r0 * math.sin(a),
                       cx + r1 * math.cos(a), cy + r1 * math.sin(a),
                       0.5 + 2.6 * p, (255, 196, 130, int(20 + 120 * p)))
        for sx, sy, rr, ph in [(390, 78, 24, 0.0), (66, 208, 15, 0.45),
                               (72, 318, 13, 0.75), (398, 120, 10, 0.25)]:
            v = 0.25 + 0.75 * R.beat(t, 2.0, ph)   # 原画已有星光的柔光呼吸
            fx.glow_blob(sx, sy, rr * (0.5 + 0.5 * v), (255, 226, 170, int(18 + 58 * v)))
        fx.glow_blob(430, 300, 24, (255, 120, 46, int(22 + 20 * sway)))
        out = R.over(out, fx.finish(bloom=0.5, bloom_sigma=5.0))
        out = R.camera(out, 1.0 + 0.006 * p, 0, 0)
        out = R.over(out, R.name_chip(sp.chip))
        return np.clip(out, 0, 1)

    if "--debug" in sys.argv:
        debug_overlay(img, [(arm_mask, (255, 60, 60)), (eyes.mask, (60, 255, 120))],
                      DEBUG / "01_masks.png")
    return sp, frame


# ================================================================ 02 思考
def build_02():
    sp = SPECS[1]
    img = load("02_思考")

    hand_poly = [(136, 302), (148, 278), (170, 270), (196, 278), (213, 296), (219, 320),
                 (214, 346), (231, 376), (239, 406), (220, 426), (194, 420), (168, 404),
                 (151, 379), (139, 349)]
    hand_mask = R.poly_mask(hand_poly, feather=2.0)
    hand = R.extract(img, hand_mask)

    scrib_mask = R.ellipse_mask(85, 255, 36, 36, feather=2.0)
    scrib = R.extract(img, scrib_mask)

    eyes = R.ScreenEyes(img, quad=[(148, 202), (336, 190), (342, 308), (150, 318)], thr=0.66)
    base = eyes.plate

    dx_keys = [(0.00, 0.0), (0.22, -3.0), (0.44, 2.5), (0.62, 6.0), (0.86, 0.0), (1.0, 0.0)]
    dy_keys = [(0.00, 0.0), (0.22, -2.0), (0.44, -4.0), (0.62, -2.5), (0.86, 0.0), (1.0, 0.0)]

    def frame(i):
        t = i / sp.frames
        tap = R.beat(t, 2.0) ** 0.6
        blink = R.ease_pulse(t, 0.80, 0.05)          # 想太久，眨一下
        aha = R.ease_pulse(t, 0.55, 0.22)            # 灯泡亮起来的顿悟时刻

        dx, dy = R.key_curve(t, dx_keys), R.key_curve(t, dy_keys)
        out = eyes.render(t, scale=1.0 + 0.012 + 0.010 * tap, dx=dx, dy=dy,
                          blink=blink, glow=0.10 + 0.35 * aha)

        # 手：以手腕为轴轻微摆动（配合 1.04 放大，放大后仍覆盖原位，不会露底板）
        out = R.over(out, R.warp_layer(hand, R.affine((226, 410), 1.6 * math.sin(2 * math.pi * t),
                                                     1.0 + 0.030)))
        # 涂鸦：绕自身中心缓慢转动
        out = R.over(out, R.warp_layer(scrib, R.affine((85, 255), 7.0 * math.sin(2 * math.pi * t),
                                                       1.06)))

        fx = R.FX()
        # 灯泡光晕 + 放射细线
        br = 1.0 if aha > 0 else 0.25
        v = 0.30 + 0.70 * aha
        fx.glow_blob(414, 248, 26, (255, 236, 170, int(12 + 70 * v)))
        if aha > 0.15:
            for k in range(9):
                a = k * 2 * math.pi / 9 + 0.15 * math.sin(2 * math.pi * t)
                r0, r1 = 26, 30 + 20 * aha
                fx.tapered(414 + r0 * math.cos(a), 248 + r0 * math.sin(a),
                           414 + r1 * math.cos(a), 248 + r1 * math.sin(a),
                           1.4, (255, 232, 150, int(70 * aha)))
        # 咖啡热气：两条缓慢上升的细线
        for k in range(2):
            ph = t + k * 0.5
            pts = []
            for s in range(9):
                u = s / 8
                y = 388 - u * 46
                x = 158 + k * 14 + 5 * math.sin(2 * math.pi * (u * 1.4 + ph)) * u
                pts.append((x, y))
            fx.line(pts, (255, 244, 226, 26 + int(16 * (0.5 + 0.5 * math.sin(2 * math.pi * ph)))),
                    width=2.4)
        out = R.over(out, fx.finish(bloom=0.45, bloom_sigma=5.0))
        out = R.camera(out, 1.0 + 0.004 * math.sin(2 * math.pi * t))
        out = R.over(out, R.name_chip(sp.chip))
        return np.clip(out, 0, 1)

    if "--debug" in sys.argv:
        debug_overlay(img, [(hand_mask, (255, 80, 60)), (scrib_mask, (255, 220, 60)),
                            (eyes.mask, (60, 255, 120))], DEBUG / "02_masks.png")
    return sp, frame


# ================================================================ 03 震惊
def build_03():
    sp = SPECS[2]
    img = load("03_震惊")

    papers = [
        ([(26, 16), (104, 6), (122, 50), (44, 74)], (75, 44), -1.0),
        ([(450, 148), (498, 128), (512, 200), (468, 230)], (480, 185), 1.0),
        ([(48, 380), (108, 358), (136, 438), (70, 462)], (88, 404), -1.0),
    ]
    layers = []
    for poly, cen, direction in papers:
        m = R.poly_mask(poly, feather=2.0)
        layers.append((R.extract(img, m), cen, direction, m))

    eyes = R.ScreenEyes(img, quad=[(165, 128), (335, 112), (340, 250), (170, 265)], thr=0.70)
    base = eyes.plate

    def frame(i):
        t = i / sp.frames
        # 冲击放在循环中段，避开首尾接缝，循环时不会在接缝处突然抖一下
        hit = R.ease_pulse(t, 0.28, 0.20)
        settle = R.ease_pulse(t, 0.62, 0.26)
        jitter = math.sin(2 * math.pi * t * 6) * hit

        out = eyes.render(t, scale=1.0 + 0.105 * hit, dx=1.6 * jitter, dy=-1.2 * jitter,
                          glow=0.18 + 0.55 * hit, dim=1.0 - 0.06 * hit)

        # 纸片：各自绕中心旋转并轻微放大（放大保证覆盖原位，不留残影）
        for (lay, cen, direction, _), ph in zip(layers, (0.0, 0.37, 0.71)):
            ang = direction * (3.6 * math.sin(2 * math.pi * (t + ph))
                               + 5.0 * hit * math.sin(4 * math.pi * (t + ph)))
            out = R.over(out, R.warp_layer(lay, R.affine(cen, ang, 1.0 + 0.11 * hit)))

        fx = R.FX()
        # 冲击表现：柔和闪光 + 只在画面四周边框出现的漫画集中线
        if hit > 0.02:
            fx.glow_blob(252, 168, 96 + 30 * hit, (255, 214, 168, int(70 * hit)))
            cx, cy = 256.0, 205.0
            for k in range(14):
                a = 2 * math.pi * k / 14 + 0.12
                dxa, dya = math.cos(a), math.sin(a)
                tx = ((R.S - cx) / dxa) if dxa > 0 else ((0.0 - cx) / dxa if dxa < 0 else 1e9)
                ty = ((R.S - cy) / dya) if dya > 0 else ((0.0 - cy) / dya if dya < 0 else 1e9)
                t_edge = min(tx, ty) * 0.995
                sx, sy = cx + t_edge * dxa, cy + t_edge * dya
                ln = 46 + 62 * hit
                fx.tapered(sx, sy, sx - ln * dxa, sy - ln * dya,
                           8.0 + 8.0 * hit, (255, 250, 240, int(78 * hit)))
        # 惊吓火星（只在四角，属于画面边缘的空气感）
        for k, ph in enumerate((0.0, 0.33, 0.66)):
            v = 0.2 + 0.8 * R.beat(t, 2.0, ph)
            fx.glow_blob(96 + k * 168, 84 + (k % 2) * 34, 9 + 5 * v,
                         (255, 196, 120, int(30 + 70 * v)))
        out = R.over(out, fx.finish(bloom=0.55, bloom_sigma=6.0))

        # 整体后仰 + 抖动（幅度小、时间短，表现为被吓一跳而不是整图乱晃）
        out = R.camera(out, 1.0 + 0.022 * hit + 0.004 * settle,
                       2.2 * jitter, -1.4 * hit)
        out = R.over(out, R.name_chip(sp.chip))
        return np.clip(out, 0, 1)

    if "--debug" in sys.argv:
        debug_overlay(img, [(m, (255, 80, 60)) for _, _, _, m in layers]
                      + [(eyes.mask, (60, 255, 120))], DEBUG / "03_masks.png")
    return sp, frame


# ================================================================ 04 累瘫
def build_04():
    sp = SPECS[3]
    img = load("04_累瘫")

    head_poly = [(232, 194), (300, 178), (372, 206), (430, 250), (446, 320),
                 (420, 380), (350, 412), (268, 408), (200, 372), (178, 306), (190, 250)]
    eyes = R.ScreenEyes(img, quad=[(190, 236), (368, 230), (376, 350), (192, 368)], thr=0.60)
    head_mask = R.poly_mask(head_poly, feather=3.0)
    # 头部图层取自"已去掉眼睛"的图，眼睛由 ScreenEyes 单独驱动，避免两套眼睛叠在一起
    head = R.extract(eyes.plate, head_mask)
    # 头部压缩时顶边会下移，露出的一条窄缝用 inpaint 补
    base = R.inpaint_plate(eyes.plate, head_mask, grow=3, radius=9)

    scrib_mask = R.ellipse_mask(420, 94, 52, 42, feather=2.0)
    scrib = R.extract(img, scrib_mask)

    piv = (300, 408)   # 头与桌面的接触点

    def frame(i):
        t = i / sp.frames
        breath = 0.5 - 0.5 * math.cos(2 * math.pi * t)      # 一次很慢的呼吸
        droop = math.sin(2 * math.pi * t) * 0.5 + 0.5
        off = R.key_curve(t, [(0.00, 1.0), (0.22, 1.0), (0.34, 0.18), (0.60, 0.25),
                              (0.72, 0.85), (0.86, 1.0), (1.0, 1.0)])
        flick = 1.0 if not (0.34 < t < 0.44) else 0.55 + 0.45 * abs(math.sin(2 * math.pi * t * 9))

        M_head = np.array(
            [[1.0 + 0.006 * breath, 0.0, -0.006 * breath * piv[0]],
             [0.0, 1.0 - 0.016 * breath, 0.016 * breath * piv[1] + 5.0 * breath]], np.float32)
        out = R.over(base, R.warp_layer(head, M_head))
        # 眼睛跟随头部同一个变换，再叠加"断电"变暗
        out = eyes.render(t, base=out, matrix=M_head, dim=off * flick, glow=0.05 * off)
        out = R.over(out, R.warp_layer(scrib, R.affine((420, 94), 9.0 * math.sin(2 * math.pi * t),
                                                       1.06)))

        fx = R.FX()
        # 热气：从头顶两处缓慢冒出
        for k, (bx, by) in enumerate([(250, 200), (330, 190)]):
            ph = t + k * 0.45
            pts = []
            for s in range(10):
                u = s / 9
                y = by - u * 54
                x = bx + 9 * math.sin(2 * math.pi * (u * 1.2 + ph)) * u + 4 * k
                pts.append((x, y))
            fx.line(pts, (240, 226, 214, 18 + int(12 * (0.5 + 0.5 * math.sin(2 * math.pi * ph)))),
                    width=3.0)
        # 低电量红光亮灭
        bat = 0.35 + 0.65 * abs(math.sin(2 * math.pi * t * 1.5))
        fx.glow_blob(96, 198, 22, (255, 70, 60, int(20 + 46 * bat)))
        fx.glow_blob(300, 300, 90, (60, 70, 90, int(10 + 8 * droop)))
        out = R.over(out, fx.finish(bloom=0.45, bloom_sigma=6.0))
        out = R.camera(out, 1.0 + 0.005 * breath, 0, 1.2 * breath)
        out = R.over(out, R.name_chip(sp.chip))
        return np.clip(out, 0, 1)

    if "--debug" in sys.argv:
        debug_overlay(img, [(head_mask, (255, 80, 60)), (eyes.mask, (60, 255, 120)),
                            (scrib_mask, (255, 220, 60))], DEBUG / "04_masks.png")
        R.save_rgb(base, DEBUG / "04_plate.png")
    return sp, frame


# ================================================================ 05 冲刺
def build_05():
    sp = SPECS[4]
    img = load("05_冲刺")

    eyes = R.ScreenEyes(img, quad=[(258, 60), (440, 88), (420, 160), (255, 143)], thr=0.70)
    base = eyes.plate

    cape_mask = R.poly_mask([(46, 52), (108, 44), (170, 78), (212, 132), (204, 180),
                             (160, 202), (104, 178), (58, 130), (40, 88)], feather=10.0)
    win = R.bbox_window(cape_mask) * cape_mask

    def frame(i):
        t = i / sp.frames
        stride = R.beat(t, 2.0)
        run = 0.5 - 0.5 * math.cos(4 * math.pi * t)

        out = eyes.render(t, scale=1.0 + 0.028 * stride, dx=2.6 * stride, dy=0.0,
                          glow=0.12 + 0.30 * stride)

        # 披风：边界钉住的位移场，横向行波，边缘位移为 0 所以不会裂开
        amp = 13.0 * (0.35 + 0.65 * stride)
        yy = np.arange(R.S, dtype=np.float32)[:, None] / 44.0
        xx = np.arange(R.S, dtype=np.float32)[None, :] / 96.0
        phase = 2 * math.pi * t
        dy = amp * win * np.sin(yy * 2.0 + phase) * 0.75
        dx = amp * win * np.cos(yy * 1.3 + xx + phase) * 0.45
        out = R.warp_field(out, dx.astype(np.float32), dy.astype(np.float32))

        fx = R.FX()
        # 脚底冲击：向外抛出的尘土短划
        for k in range(12):
            a = math.pi * (0.08 + 0.84 * k / 11)
            r0 = 92 + 42 * ((t * 2 + k * 0.29) % 1.0)
            r1 = r0 + 30 + 54 * stride
            fx.tapered(262 + r0 * math.cos(a), 458 - r0 * 0.30 * math.sin(a),
                       262 + r1 * math.cos(a), 458 - r1 * 0.30 * math.sin(a),
                       2.0 + 2.4 * stride, (255, 232, 200, int(30 + 96 * stride)))
        # 两侧掠过的速度线（左侧进、右侧出，强化迎面冲刺）
        for k in range(8):
            y = 40 + k * 60 + 6 * math.sin(2 * math.pi * t * 2 + k)
            off = (t * 2.2 + k * 0.11) % 1.0
            x0 = -80 + off * 70
            fx.tapered(x0, y, x0 + 64 + 34 * run, y - 6, 1.8,
                       (255, 255, 255, int(26 + 40 * run)))
        for k in range(4):                       # 右下角掠出
            y = 320 + k * 48
            off = (t * 2.2 + k * 0.21) % 1.0
            x0 = 500 + off * 40
            fx.tapered(x0, y, x0 - 58 - 26 * run, y + 8, 1.8,
                       (255, 244, 226, int(34 + 34 * run)))
        out = R.over(out, fx.finish(bloom=0.4, bloom_sigma=5.0))

        # 跑步节拍：轻推近 + 上下冲击（每秒 2 步的跑步节奏）
        out = R.camera(out, 1.0 + 0.014 * run, 1.0 * math.sin(2 * math.pi * t * 3),
                       -3.0 * stride)
        out = R.over(out, R.name_chip(sp.chip))
        return np.clip(out, 0, 1)

    if "--debug" in sys.argv:
        debug_overlay(img, [(cape_mask, (255, 60, 60)), (eyes.mask, (60, 255, 120))],
                      DEBUG / "05_masks.png")
    return sp, frame


# ================================================================ 06 辛苦啦
def build_06():
    sp = SPECS[5]
    img = load("06_辛苦啦")

    eyes = R.ScreenEyes(img, quad=[(203, 150), (380, 158), (382, 242), (205, 244)], thr=0.66)
    base = eyes.plate

    # 关于像素爱心为什么不做几何放大（实测结论）：
    # 爱心被两只深色手套左右夹住，背景又是同一种高饱和暖红，颜色阈值和多边形都不够干净；
    # 试过「多边形 ∩ 红色/白色像素」+ 手套静止层，放大后仍会留下重影手掌和细弧线；
    # 因此不做几何缩放，改用光晕跳动 + 描边爱心上浮来表达"送心"。
    heart_c = (312, 340)


    def frame(i):
        t = i / sp.frames
        # 心跳：一个循环里两次"咚—咚"，第二次更轻
        beat1 = R.ease_pulse(t, 0.16, 0.16)
        beat2 = R.ease_pulse(t, 0.34, 0.13) * 0.6
        push = beat1 + beat2
        breathe = 0.5 - 0.5 * math.cos(2 * math.pi * t)     # 持续呼吸，保证整段都在动
        glow = 0.35 + 0.65 * push

        out = eyes.render(t, scale=1.0 + 0.020 + 0.016 * breathe + 0.036 * push,
                          glow=0.12 + 0.10 * breathe + 0.32 * push)
        # 夕阳下屏幕玻璃缓慢扫过一道反光：给整段提供持续、低幅度的运动
        out = R.moving_glare(out, [(203, 150), (380, 158), (382, 242), (205, 244)],
                             (t * 1.0) % 1.0, alpha=0.12)

        fx = R.FX()
        # 心跳：爱心区域的光晕随节拍鼓起，代替无法干净分离的几何缩放
        fx.glow_blob(heart_c[0], heart_c[1], 74 + 26 * push,
                     (255, 96, 96, int(16 + 66 * push)))
        fx.glow_blob(heart_c[0], heart_c[1], 128, (255, 150, 110, int(6 + 10 * breathe)))
        # 从爱心向上飘出的描边爱心
        for k in range(7):
            ph = (t * 1.0 + k / 7.0) % 1.0
            x = 215 + (k * 41) % 160 + 12 * math.sin(2 * math.pi * ph + k)
            y = 316 - ph * 244
            alpha = int(200 * (1.0 - ph) ** 1.2)
            size = 8 + 4 * (k % 3)
            fx.heart(x, y, size, (255, 178, 196, alpha))
            fx.heart(x, y, size, (255, 255, 255, min(255, alpha + 30)), outline=True)
        # 暖光呼吸
        fx.glow_blob(300, 330, 120, (255, 150, 110, int(8 + 10 * breathe + 8 * push)))
        out = R.over(out, fx.finish(bloom=0.55, bloom_sigma=7.0))
        out = R.camera(out, 1.0 + 0.004 * push, 0, -1.0 * push)
        out = R.over(out, R.name_chip(sp.chip))
        return np.clip(out, 0, 1)

    if "--debug" in sys.argv:
        debug_overlay(img, [(eyes.mask, (60, 255, 120))], DEBUG / "06_masks.png")
    return sp, frame


BUILDERS = {"01": build_01, "02": build_02, "03": build_03,
            "04": build_04, "05": build_05, "06": build_06}


# ---------------------------------------------------------------- 导出

def render_sticker(sid):
    sp, frame = BUILDERS[sid]()
    frames = [frame(i) for i in range(sp.frames)]
    for d in (OUT_GIF, OUT_WEBP, OUT_PNG):
        d.mkdir(exist_ok=True)

    hero = frames[min(sp.frames - 1, int(round(sp.frames * sp.hero)))]
    R.save_rgb(hero, OUT_PNG / f"{sp.sid}_{sp.name}.png")

    gs = R.export_gif(frames, OUT_GIF / f"{sp.sid}_{sp.name}.gif", sp.duration)
    ws = R.export_webp(frames, OUT_WEBP / f"{sp.sid}_{sp.name}.webp", sp.duration)
    print(f"{sp.sid} {sp.name}: {sp.frames}帧 {sp.frames * sp.duration / 1000:.2f}s "
          f"GIF {gs / 1048576:.2f}MiB WEBP {ws / 1048576:.2f}MiB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    ids = [args.only] if args.only else [s.sid for s in SPECS]
    for sid in ids:
        render_sticker(sid)


if __name__ == "__main__":
    main()
