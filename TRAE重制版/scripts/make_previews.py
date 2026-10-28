"""
生成展示用预览：
  · 预览_六宫格.png          —— 六张重制版的成组展示
  · 对照_旧版与重制.png       —— 上排 v5 原画 / 下排 重制版动作峰值帧
  · 过程截图/分镜_XX.png      —— 每张取 6 个时间点，展示动作过程（真实渲染帧）
  · 过程截图/遮罩_XX.png      —— 构建时输出的分层遮罩核对图（需先跑 --debug）

用法：python scripts/build_remaster.py --debug && python scripts/make_previews.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trae_rig as R
from build_remaster import SPECS

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT.parent / "TRAE宝表情包v5"
SHOT = ROOT / "过程截图"
DEBUG = Path("/tmp/trae_build/debug")

ACTIONS = {
    "01": "出拳两次：手臂以肩为支点放大上抬，配合速度线与星光呼吸",
    "02": "托腮思考：手部摆动、视线在屏幕内游移、灯泡亮起、涂鸦旋转",
    "03": "受惊后仰：整体轻推近+短促抖动，纸片各自旋转，眼睛放大发光",
    "04": "力竭下沉：头部以桌面接触点为轴压缩下沉，屏幕亮度掉到低电量",
    "05": "迎面冲刺：披风行波形变、脚下扬尘、两侧速度线、跑步节拍推进",
    "06": "送心：爱心光晕随心跳鼓起，描边爱心持续上浮，眼睛同步心跳",
}


def font(size):
    return R._font(size)


def card(img, w, label, sub, fg=(240, 244, 248)):
    card_im = Image.new("RGB", (w, w + 58), (22, 24, 30))
    im = img.resize((w, w), Image.LANCZOS)
    card_im.paste(im, (0, 0))
    d = ImageDraw.Draw(card_im)
    d.text((14, w + 9), label, font=font(24), fill=fg)
    d.text((14, w + 37), sub, font=font(15), fill=(150, 160, 175))
    return card_im


def six_grid():
    W, GAP, PAD, TOP = 460, 18, 28, 96
    cols, rows = 3, 2
    width = PAD * 2 + cols * W + (cols - 1) * GAP
    height = TOP + rows * (W + 58) + (rows - 1) * GAP + PAD
    sheet = Image.new("RGB", (width, height), (16, 17, 22))
    d = ImageDraw.Draw(sheet)
    d.text((PAD, 26), "TRAE宝 · 六连发表情包", font=font(40), fill=(248, 250, 252))
    d.text((PAD, 70), "重制版  ·  512×512  ·  循环动态  ·  GIF / WebP",
           font=font(18), fill=(255, 150, 92))
    for i, sp in enumerate(SPECS):
        im = Image.open(ROOT / "png" / f"{sp.sid}_{sp.name}.png").convert("RGB")
        c = card(im, W, f"{sp.sid}  {sp.name}", ACTIONS[sp.sid])
        x = PAD + (i % cols) * (W + GAP)
        y = TOP + (i // cols) * (W + 58 + GAP)
        sheet.paste(c, (x, y))
    sheet.save(ROOT / "预览_六宫格.png", optimize=True)
    print("预览_六宫格.png", sheet.size)


def compare_sheet():
    W, GAP, PAD, TOP = 238, 12, 24, 88
    width = PAD * 2 + 6 * W + 5 * GAP
    height = TOP + 2 * (W + 34) + GAP + PAD
    sheet = Image.new("RGB", (width, height), (16, 17, 22))
    d = ImageDraw.Draw(sheet)
    d.text((PAD, 22), "旧版 v5 原画  vs  TRAE 重制版", font=font(32), fill=(248, 250, 252))
    d.text((PAD, 62), "上排：用户已有的 ChatGPT 生成原画（仅作视觉基准，未改动）     "
                      "下排：本次分层动画的动作峰值帧",
           font=font(15), fill=(150, 160, 175))
    for i, sp in enumerate(SPECS):
        old = Image.open(V5 / f"{sp.sid}_{sp.name}_原画.png").convert("RGB").resize((W, W), Image.LANCZOS)
        new = Image.open(ROOT / "png" / f"{sp.sid}_{sp.name}.png").convert("RGB").resize((W, W), Image.LANCZOS)
        x = PAD + i * (W + GAP)
        sheet.paste(old, (x, TOP))
        sheet.paste(new, (x, TOP + W + 34 + GAP))
        d.text((x, TOP + W + 7), f"{sp.sid}  旧版", font=font(15), fill=(150, 160, 175))
        d.text((x, TOP + 2 * W + 34 + GAP + 6), f"{sp.sid}  重制版", font=font(15), fill=(255, 160, 100))
    sheet.save(ROOT / "对照_旧版与重制.png", optimize=True)
    print("对照_旧版与重制.png", sheet.size)


def strips():
    for sp in SPECS:
        gif = Image.open(ROOT / "gif" / f"{sp.sid}_{sp.name}.gif")
        n = gif.n_frames
        picks = [round(n * f) % n for f in (0, 0.17, 0.33, 0.5, 0.67, 0.84)]
        W, GAP, PAD, TOP = 240, 10, 16, 70
        sheet = Image.new("RGB", (PAD * 2 + 3 * W + 2 * GAP, TOP + 2 * W + GAP + PAD + 26),
                          (16, 17, 22))
        d = ImageDraw.Draw(sheet)
        d.text((PAD, 16), f"{sp.sid} {sp.name} · 动作分镜（6 / {n} 帧）",
               font=font(22), fill=(240, 244, 248))
        d.text((PAD, 44), ACTIONS[sp.sid], font=font(13), fill=(150, 160, 175))
        for idx, fi in enumerate(picks):
            gif.seek(fi)
            im = gif.convert("RGB").resize((W, W), Image.LANCZOS)
            x = PAD + (idx % 3) * (W + GAP)
            y = TOP + (idx // 3) * (W + GAP)
            sheet.paste(im, (x, y))
            d.text((x + 6, y + 6), f"f{fi}", font=font(14), fill=(255, 220, 120))
        sheet.save(SHOT / f"分镜_{sp.sid}_{sp.name}.png", optimize=True)
    print("分镜 6 张")


def copy_masks():
    n = 0
    for sp in SPECS:
        src = DEBUG / f"{sp.sid}_masks.png"
        if src.exists():
            shutil.copy2(src, SHOT / f"遮罩_{sp.sid}_{sp.name}.png")
            n += 1
    print(f"遮罩核对图 {n} 张")


if __name__ == "__main__":
    SHOT.mkdir(exist_ok=True)
    six_grid()
    compare_sheet()
    strips()
    copy_masks()
