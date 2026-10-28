"""
输出质检：尺寸 / 帧数 / 时长 / 循环 / 体积 / 帧间变化 / 首尾衔接。
用法：python scripts/verify_outputs.py
结果同时写入 ../检查报告.json 与终端。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_remaster import SPECS  # noqa: E402

EXPECT_DUR = {s.sid: s.duration * s.frames for s in SPECS}


def frames_of(im):
    out = []
    for i in range(im.n_frames):
        im.seek(i)
        out.append(np.asarray(im.convert("RGB")).astype(np.int16))
    return out


def webp_durations(path):
    """Pillow 的 WebP 读取器不返回 duration，这里直接解析 ANMF 块。"""
    data = Path(path).read_bytes()
    i, durs = 12, []
    while i + 8 <= len(data):
        fourcc = data[i:i + 4]
        size = int.from_bytes(data[i + 4:i + 8], "little")
        if fourcc == b"ANMF":
            payload = data[i + 8:i + 24]
            if len(payload) >= 15:
                durs.append(int.from_bytes(payload[12:15], "little"))
        i += 8 + size + (size % 2)
    return durs


def check_one(spec):
    gif = ROOT / "gif" / f"{spec.sid}_{spec.name}.gif"
    webp = ROOT / "webp" / f"{spec.sid}_{spec.name}.webp"
    png = ROOT / "png" / f"{spec.sid}_{spec.name}.png"
    rec = {"id": spec.sid, "name": spec.name}
    problems = []

    g = Image.open(gif)
    pal_colors = len(g.getpalette() or []) // 3

    durs = []
    for i in range(g.n_frames):
        g.seek(i)
        durs.append(g.info.get("duration", 0))
    fr = frames_of(g)
    rec["gif"] = {
        "size": list(g.size), "bytes": gif.stat().st_size, "frames": g.n_frames,
        "duration_ms": sum(durs), "per_frame_ms": sorted(set(durs)),
        "loop": g.info.get("loop"), "palette_colors": pal_colors,
    }
    if g.size != (512, 512):
        problems.append("GIF 尺寸不是 512x512")
    if g.n_frames != spec.frames:
        problems.append(f"GIF 帧数 {g.n_frames} != 设计 {spec.frames}")
    if sum(durs) != EXPECT_DUR[spec.sid]:
        problems.append(f"GIF 总时长 {sum(durs)}ms != 设计 {EXPECT_DUR[spec.sid]}ms")
    if g.info.get("loop") != 0:
        problems.append("GIF 未设置无限循环")

    a = np.stack([f.astype(np.float32) for f in fr])

    def mean_diff(x, y):
        return float(np.abs(x - y).mean())

    def changed_ratio(x, y, thr=4):
        """有明显变化的像素占比，比全图均值更能反映“是否在动”。"""
        return float((np.abs(x - y).max(axis=2) > thr).mean())

    diffs = [mean_diff(a[i + 1], a[i]) for i in range(len(a) - 1)]
    ratios = [changed_ratio(a[i + 1], a[i]) for i in range(len(a) - 1)]
    seam = mean_diff(a[0], a[-1])
    seam_ratio = changed_ratio(a[0], a[-1])
    rec["motion"] = {
        "min_consecutive_diff": round(min(diffs), 3),
        "max_consecutive_diff": round(max(diffs), 3),
        "mean_consecutive_diff": round(float(np.mean(diffs)), 3),
        "min_changed_pixel_ratio": round(min(ratios), 4),
        "mean_changed_pixel_ratio": round(float(np.mean(ratios)), 4),
        "loop_seam_diff_first_vs_last": round(seam, 3),
        "loop_seam_changed_ratio": round(seam_ratio, 4),
    }
    if min(ratios) < 0.001:
        problems.append(f"存在几乎静止的相邻帧（变化像素占比 {min(ratios)*100:.2f}%）")
    if seam > max(2.5, float(np.mean(diffs)) * 1.6) and seam_ratio > 0.06:
        problems.append(f"首尾衔接差异偏大（均值 {seam:.2f} / 占比 {seam_ratio*100:.1f}%），循环可能跳帧")

    w = Image.open(webp)
    wd = webp_durations(webp)
    rec["webp"] = {"size": list(w.size), "bytes": webp.stat().st_size,
                   "frames": w.n_frames, "duration_ms": sum(wd),
                   "per_frame_ms": sorted(set(wd)), "loop": w.info.get("loop")}
    if w.n_frames != spec.frames:
        problems.append(f"WebP 帧数 {w.n_frames} != 设计 {spec.frames}")
    if len(wd) != spec.frames:
        problems.append(f"WebP 时长块 {len(wd)} 个 != 设计 {spec.frames}")
    if sum(wd) != EXPECT_DUR[spec.sid]:
        problems.append(f"WebP 总时长 {sum(wd)}ms != 设计 {EXPECT_DUR[spec.sid]}ms")

    p = Image.open(png)
    rec["png"] = {"size": list(p.size), "bytes": png.stat().st_size}
    if p.size != (512, 512):
        problems.append("PNG 预览尺寸不是 512x512")

    rec["problems"] = problems
    return rec


def main():
    report = []
    for spec in SPECS:
        r = check_one(spec)
        report.append(r)
        g = r["gif"]
        print(f'{r["id"]} {r["name"]:5s} GIF {g["size"][0]}x{g["size"][1]} '
              f'{g["frames"]}帧 {g["duration_ms"]}ms {g["bytes"]/1048576:.2f}MiB | '
              f'帧变化占比 min {r["motion"]["min_changed_pixel_ratio"]*100:.2f}% '
              f'mean {r["motion"]["mean_changed_pixel_ratio"]*100:.1f}% | '
              f'首尾 {r["motion"]["loop_seam_diff_first_vs_last"]:.2f}/'
              f'{r["motion"]["loop_seam_changed_ratio"]*100:.1f}% | '
              f'WEBP {r["webp"]["bytes"]/1048576:.2f}MiB | '
              + ("OK" if not r["problems"] else "问题: " + "; ".join(r["problems"])))
    dest = ROOT / "检查报告.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    bad = [r for r in report if r["problems"]]
    print(f"\n共 {len(report)} 张，异常 {len(bad)} 张。报告：{dest}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
