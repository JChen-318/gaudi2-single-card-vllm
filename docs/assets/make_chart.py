"""Generate the README chart (light and dark variants): decode round time vs context length, single stream.

Data: measured on one Gaudi2, Qwen3.8-27B-FP8 + DFlash2, non-greedy sampling. See docs/OPTIMIZATION.md.
Usage: python docs/assets/make_chart.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# (prompt tokens in thousands, ms per speculative round)
BEFORE = [(0.62, 44.8), (7.69, 50.2), (30.47, 63.4), (91.26, 292.2)]
AFTER = [(0.62, 42.3), (7.69, 45.0), (30.47, 49.5), (57.09, 57.1), (85.59, 62.6)]
THEMES = {
    "light": dict(text="#1f2328", muted="#59636e", grid="#d1d9e0", surface="#ffffff", before="#2a78d6", after="#eb6834"),
    "dark": dict(text="#f0f6fc", muted="#9198a1", grid="#3d444d", surface="#0d1117", before="#3987e5", after="#d95926"),
}
W, H = 760, 380
X0, X1, Y0, Y1 = 64, 624, 316, 72          # plot box: x from 0 to 100K, y from 0 to 300 ms
FONT = "system-ui, -apple-system, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', 'Noto Sans CJK SC', sans-serif"


def px(k):
    return X0 + k / 100 * (X1 - X0)


def py(ms):
    return Y0 - ms / 300 * (Y0 - Y1)


def series(points, color, surface):
    path = " ".join(f"{'M' if i == 0 else 'L'}{px(k):.1f},{py(ms):.1f}" for i, (k, ms) in enumerate(points))
    out = [f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>']
    out += [f'<circle cx="{px(k):.1f}" cy="{py(ms):.1f}" r="4" fill="{color}" stroke="{surface}" stroke-width="2"/>'
            for k, ms in points]
    return out


for name, c in THEMES.items():
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
         f'aria-label="每轮解码耗时随上下文长度的变化：优化前在 91K 上下文时为 292 毫秒，优化后在 86K 时为 63 毫秒">',
         f'<g font-family="{FONT}">',
         f'<text x="{X0}" y="26" font-size="15" font-weight="600" fill="{c["text"]}">每轮解码耗时 vs 上下文长度</text>',
         f'<text x="{X0}" y="46" font-size="12" fill="{c["muted"]}">单路，毫秒 / 轮，越低越好</text>']
    # legend (identity is carried by the line key beside the text, text stays in text colours)
    lx = 330
    for label, color in (("优化前（v2）", c["before"]), ("优化后（v4+v5）", c["after"])):
        s.append(f'<line x1="{lx}" y1="41" x2="{lx + 22}" y2="41" stroke="{color}" stroke-width="2" stroke-linecap="round"/>')
        s.append(f'<circle cx="{lx + 11}" cy="41" r="4" fill="{color}" stroke="{c["surface"]}" stroke-width="2"/>')
        s.append(f'<text x="{lx + 30}" y="45" font-size="12" fill="{c["text"]}">{label}</text>')
        lx += 138
    # grid and axes
    for ms in (0, 100, 200, 300):
        s.append(f'<line x1="{X0}" y1="{py(ms):.1f}" x2="{X1}" y2="{py(ms):.1f}" stroke="{c["grid"]}" stroke-width="1"/>')
        s.append(f'<text x="{X0 - 10}" y="{py(ms) + 4:.1f}" font-size="12" text-anchor="end" fill="{c["muted"]}">{ms}</text>')
    for k in (0, 20, 40, 60, 80, 100):
        s.append(f'<text x="{px(k):.1f}" y="{Y0 + 20}" font-size="12" text-anchor="middle" fill="{c["muted"]}">'
                 f'{"0" if k == 0 else f"{k}K"}</text>')
    s.append(f'<text x="{(X0 + X1) / 2:.1f}" y="{Y0 + 44}" font-size="12" text-anchor="middle" fill="{c["muted"]}">提示词长度（token）</text>')
    # the old warm-up bucket ceiling
    s.append(f'<line x1="{px(65.5):.1f}" y1="{Y1}" x2="{px(65.5):.1f}" y2="{Y0}" stroke="{c["grid"]}" stroke-width="1"/>')
    s.append(f'<text x="{px(65.5) - 8:.1f}" y="{Y1 + 14}" font-size="11" text-anchor="end" fill="{c["muted"]}">旧预热桶上限 65.5K</text>')
    s += series(BEFORE, c["before"], c["surface"])
    s += series(AFTER, c["after"], c["surface"])
    # selective direct labels: the two end points
    k, ms = BEFORE[-1]
    s.append(f'<text x="{px(k) + 12:.1f}" y="{py(ms) + 4:.1f}" font-size="12" font-weight="600" fill="{c["text"]}">292 ms</text>')
    s.append(f'<text x="{px(k) + 12:.1f}" y="{py(ms) + 20:.1f}" font-size="11" fill="{c["muted"]}">9.8 tok/s</text>')
    k, ms = AFTER[-1]
    s.append(f'<text x="{px(k) + 12:.1f}" y="{py(ms) - 12:.1f}" font-size="12" font-weight="600" fill="{c["text"]}">63 ms</text>')
    s.append(f'<text x="{px(k) + 12:.1f}" y="{py(ms) + 4:.1f}" font-size="11" fill="{c["muted"]}">47.6 tok/s</text>')
    s += ['</g>', '</svg>']
    open(os.path.join(HERE, f"decode-round-{name}.svg"), "w", encoding="utf-8", newline="\n").write("\n".join(s) + "\n")
print("written")
