"""Architecture figure for paper.tex: grasp-type-conditioned 3D diffusion policy.

Run:  python figures/make_architecture.py
Out:  figures/architecture.pdf (vector, for LaTeX) + figures/architecture.png (preview)

Drawn at 11 x 4.4 in; include with \\includegraphics[width=\\textwidth]{figures/architecture.pdf}.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

OUT_DIR = Path(__file__).resolve().parent

# One color per information stream; the grasp-type stream (our contribution) is the warm accent.
C_PC = "#2B6CB0"
C_STATE = "#4A5568"
C_GRASP = "#DD6B20"
C_FUSE = "#6B46C1"
C_UNET = "#2C7A7B"
C_OUT = "#2F855A"
INK = "#1A202C"
MUTED = "#718096"

FS_SUB = 7.5

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 9,
})


def box(ax, x0, y0, w, h, color, title, sub=None, fill_alpha=0.10):
    for fc, ec, alpha, z in [(color, "none", fill_alpha, 1), ("none", color, 1.0, 2)]:
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=1.3, alpha=alpha, zorder=z))
    cx, cy = x0 + w / 2, y0 + h / 2
    if not title:
        return
    if sub is None:
        ax.text(cx, cy, title, ha="center", va="center", color=INK, fontweight="bold", zorder=3)
    else:
        ax.text(cx, cy + 0.13, title, ha="center", va="center", color=INK, fontweight="bold", zorder=3)
        ax.text(cx, cy - 0.16, sub, ha="center", va="center", color=MUTED, fontsize=FS_SUB, zorder=3)


def arrow(ax, p0, p1, color=INK, lw=1.2, label=None, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, ls=ls, zorder=4, shrinkA=0, shrinkB=0))
    if label:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 0.08, label,
                ha="center", va="bottom", fontsize=FS_SUB, color=color)


def line(ax, pts, color=INK, lw=1.2):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, zorder=4, solid_capstyle="round")


def brace(ax, x, y0, y1, ym, w, color=INK, lw=1.6, d=0.2):
    """Right-pointing curly brace '}' spanning y0..y1 with its tip at height ym."""
    s = x + w / 2
    verts = [(x, y1), (s, y1), (s, y1 - d), (s, ym + d), (s, ym), (x + w, ym),
             (s, ym), (s, ym - d), (s, y0 + d), (s, y0), (x, y0)]
    M, C, L = MplPath.MOVETO, MplPath.CURVE3, MplPath.LINETO
    codes = [M, C, C, L, C, C, C, C, L, C, C]
    ax.add_patch(PathPatch(MplPath(verts, codes), fc="none", ec=color, lw=lw, zorder=5,
                           capstyle="round", joinstyle="round"))


def mini_point_cloud(ax, cx, cy, rng):
    """Hold-shaped blob (half-ellipsoid), viewed obliquely from the side."""
    n = 220
    u = rng.uniform(0, np.pi, n)
    v = rng.uniform(0, np.pi / 2, n)
    x = 0.24 * np.cos(u) * np.sin(v) + 0.06 * np.cos(u) * np.sin(v) ** 2
    z = 0.22 * np.cos(v) * (0.75 + 0.25 * np.sin(3 * u))
    y = 0.10 * np.sin(u) * np.sin(v)
    ax.scatter(cx + x + 0.3 * y, cy - 0.15 + z + 0.25 * y, s=1.6, c=z, cmap="Blues",
               vmin=-0.1, vmax=0.25, lw=0, zorder=3)


def main():
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 11.1)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    rows = {"pc": 3.45, "state": 2.25, "grasp": 1.05}
    bh = 0.9
    top_label_y = 4.08

    # ---- Column A: inputs -------------------------------------------------
    ax_x, ax_w = 0.15, 2.35
    box(ax, ax_x, rows["pc"] - bh / 2, ax_w, bh, C_PC, "")
    mini_point_cloud(ax, ax_x + 0.38, rows["pc"], rng)
    ax.text(ax_x + 1.5, rows["pc"] + 0.13, "Point cloud $P$", ha="center", va="center",
            fontweight="bold", color=INK, zorder=3)
    ax.text(ax_x + 1.5, rows["pc"] - 0.16, r"$1024\times3$ XYZ, 4 cams", ha="center", va="center",
            fontsize=FS_SUB, color=MUTED, zorder=3)

    box(ax, ax_x, rows["state"] - bh / 2, ax_w, bh, C_STATE,
        r"Joint state $s_{t-1:t}$", r"$2\times23$  (arm 7 + LEAP 16)")

    gy0 = rows["grasp"] - bh / 2
    box(ax, ax_x, gy0, ax_w, bh, C_GRASP, "", fill_alpha=0.14)
    ax.text(ax_x + ax_w / 2, rows["grasp"] + 0.2, "Grasp type $g$", ha="center", va="center",
            fontweight="bold", color=INK, zorder=3)
    cw, gap = 0.5, 0.05
    x0 = ax_x + (ax_w - (4 * cw + 3 * gap)) / 2
    for i, name in enumerate(["crimp", "sloper", "pinch", "jug"]):
        on = name == "jug"  # example one-hot
        cx0 = x0 + i * (cw + gap)
        ax.add_patch(FancyBboxPatch((cx0, rows["grasp"] - 0.27), cw, 0.26,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=C_GRASP if on else "white", ec=C_GRASP, lw=0.9, zorder=3))
        ax.text(cx0 + cw / 2, rows["grasp"] - 0.14, name, ha="center", va="center", fontsize=7,
                color="white" if on else C_GRASP, fontweight="bold" if on else "normal", zorder=4)
    ax.text(ax_x + ax_w / 2, 0.2, "from VLM hold classifier or manual label",
            ha="center", va="center", fontsize=FS_SUB, color=MUTED, style="italic")
    arrow(ax, (ax_x + ax_w / 2, 0.32), (ax_x + ax_w / 2, gy0), color=MUTED, lw=1.0, ls="--")

    # ---- Column B: encoders ----------------------------------------------
    bx, bw = 2.9, 1.4
    for key, col, title, sub in [
        ("pc", C_PC, "PointNet", "MLP + max-pool"),
        ("state", C_STATE, "State MLP", None),
        ("grasp", C_GRASP, "Grasp embed", "one-hot(4) → MLP"),
    ]:
        box(ax, bx, rows[key] - bh / 2, bw, bh, col, title, sub,
            fill_alpha=0.14 if key == "grasp" else 0.10)
        arrow(ax, (ax_x + ax_w, rows[key]), (bx, rows[key]), color=col)

    # ---- Concat: brace merging all three embeddings into one vector -------
    # Outer arrows meet the brace's curled ends; the middle one meets its spine.
    kx, kw = 4.8, 0.3
    spine = kx + kw / 2
    for key, col, dim, x_end in [("pc", C_PC, "256", kx), ("state", C_STATE, "128", spine - 0.03),
                                 ("grasp", C_GRASP, "64", kx)]:
        arrow(ax, (bx + bw, rows[key]), (x_end, rows[key]), color=col, label=dim)
    brace(ax, kx, rows["grasp"], rows["pc"], rows["state"], kw, color=C_FUSE)
    ax.text(spine, rows["grasp"] - 0.12, "concat\n(448)", ha="center", va="top",
            fontsize=FS_SUB, color=C_FUSE)

    # ---- Column C: fusion MLP, noisy actions, diffusion step ---------------
    fx, fw = 5.4, 1.1
    box(ax, fx, rows["state"] - bh / 2, fw, bh, C_FUSE, "Fusion", r"MLP → $z\in\mathbb{R}^{512}$")
    arrow(ax, (kx + kw, rows["state"]), (fx, rows["state"]), color=C_FUSE, lw=2.0)
    box(ax, fx, rows["pc"] - 0.35, fw, 0.7, C_UNET, r"Noisy $a_k$", r"$a_K\sim\mathcal{N}(0,I)$")
    box(ax, fx, rows["grasp"] - 0.3, fw, 0.6, MUTED, r"Step $k$", fill_alpha=0.08)

    # ---- Column D: conditional 1D temporal U-Net --------------------------
    ux0, ux1, uy0, uy1 = 6.95, 9.05, 1.2, 3.8
    for fc, ec, alpha, z in [(C_UNET, "none", 0.07, 1), ("none", C_UNET, 1.0, 2)]:
        ax.add_patch(FancyBboxPatch((ux0, uy0), ux1 - ux0, uy1 - uy0,
                                    boxstyle="round,pad=0,rounding_size=0.1",
                                    fc=fc, ec=ec, lw=1.3, alpha=alpha, zorder=z))
    ax.text((ux0 + ux1) / 2, top_label_y, r"Temporal U-Net $\epsilon_\theta(a_k,k,z)$",
            ha="center", va="center", fontweight="bold", color=INK)

    bxs = [7.25, 7.65, 8.0, 8.35, 8.75]
    bcy = [3.1, 2.6, 2.15, 2.6, 3.1]
    bhs = [0.6, 0.45, 0.32, 0.45, 0.6]
    blk_w = 0.24
    for xc, yc, hh in zip(bxs, bcy, bhs):
        ax.add_patch(Rectangle((xc - blk_w / 2, yc - hh / 2), blk_w, hh, fc=C_UNET, ec="white",
                               lw=1, zorder=3))
    for i in range(len(bxs) - 1):
        line(ax, [(bxs[i] + blk_w / 2, bcy[i]), (bxs[i + 1] - blk_w / 2, bcy[i + 1])], color=C_UNET)
    for a, b in [(0, 4), (1, 3)]:  # skip connections
        ax.plot([bxs[a] + blk_w / 2, bxs[b] - blk_w / 2], [bcy[a], bcy[b]], color=C_UNET,
                lw=0.9, ls=(0, (2, 2)), zorder=2)
    ax.text(8.0, 3.32, "skip", ha="center", va="center", fontsize=FS_SUB, color=C_UNET)

    # noisy actions in; FiLM bus carries z + step embedding to every block
    arrow(ax, (fx + fw, 3.1), (bxs[0] - blk_w / 2, 3.1), color=C_UNET)
    bus_y, jx = 1.68, 6.72
    line(ax, [(fx + fw, rows["state"]), (jx, rows["state"]), (jx, bus_y)], color=C_FUSE)
    line(ax, [(fx + fw, rows["grasp"]), (jx, rows["grasp"]), (jx, bus_y)], color=MUTED)
    ax.add_patch(plt.Circle((jx, bus_y), 0.07, fc="white", ec=C_FUSE, lw=1.2, zorder=5))
    ax.text(jx, bus_y, "+", ha="center", va="center", fontsize=8, color=C_FUSE, zorder=6)
    line(ax, [(jx + 0.07, bus_y), (bxs[-1], bus_y)], color=C_FUSE)
    for xc, yc, hh in zip(bxs, bcy, bhs):
        arrow(ax, (xc, bus_y), (xc, yc - hh / 2), color=C_FUSE, lw=0.9)
    ax.text(8.0, bus_y - 0.13, "FiLM on every block", ha="center", va="top",
            fontsize=FS_SUB, color=C_FUSE)
    ax.text((ux0 + ux1) / 2, 0.98, r"denoise $k = K \rightarrow 1$", ha="center", va="center",
            fontsize=8, color=C_UNET, style="italic")

    # ---- Column E: action chunk + robot -----------------------------------
    # A few illustrative joint trajectories over the 16-step horizon; the first T_a steps
    # (executed before replanning) are solid, the remainder faded.
    hx0, hx1, hy0, hy1 = 9.4, 10.75, 2.2, 3.6
    T, T_A = 16, 8
    ax.add_patch(Rectangle((hx0, hy0), hx1 - hx0, hy1 - hy0, fc=C_OUT, ec="none", alpha=0.05, zorder=1))
    xa = hx0 + (hx1 - hx0) * (T_A - 0.5) / (T - 1)
    ax.add_patch(Rectangle((hx0, hy0), xa - hx0, hy1 - hy0, fc=C_OUT, ec="none", alpha=0.12, zorder=1))
    ax.plot([xa, xa], [hy0, hy1], color=C_OUT, lw=0.8, ls=(0, (2, 2)), zorder=2)
    t = np.linspace(0, 1, 120)
    tx = hx0 + (hx1 - hx0) * t
    for i, (amp, freq, ph, off) in enumerate([(0.11, 0.6, 0.2, 0.70), (0.10, 0.9, 1.6, 0.52),
                                              (0.12, 0.5, 3.0, 0.36), (0.07, 1.2, 4.2, 0.20)]):
        ty = hy0 + (hy1 - hy0) * (off + amp * np.sin(2 * np.pi * freq * t + ph))
        col = [C_OUT, "#38A169", "#68D391", "#276749"][i]
        done = tx <= xa
        ax.plot(tx[done], ty[done], color=col, lw=1.6, zorder=3, solid_capstyle="round")
        ax.plot(tx[~done], ty[~done], color=col, lw=1.2, alpha=0.35, zorder=3, solid_capstyle="round")
    ax.add_patch(Rectangle((hx0, hy0), hx1 - hx0, hy1 - hy0, fc="none", ec=C_OUT, lw=1.2, zorder=5))
    ax.text((hx0 + xa) / 2, hy0 - 0.08, r"run $T_a$", ha="center", va="top", fontsize=7,
            color=C_OUT, zorder=6)
    ax.text((hx0 + hx1) / 2, top_label_y, r"Actions $a_{t:t+15}$", ha="center", va="center",
            fontweight="bold", color=INK)
    ax.text((hx0 + hx1) / 2, hy1 + 0.1, r"16 steps $\times$ 23 joints", ha="center", va="bottom",
            fontsize=FS_SUB, color=MUTED)
    ax.text((xa + hx1) / 2, hy0 - 0.08, "time →", ha="center", va="top", fontsize=7, color=MUTED)
    arrow(ax, (bxs[-1] + blk_w / 2, 3.1), (hx0, 3.1), color=C_OUT)

    # Robot: uses figures/robot.png (transparent background, roughly square) if present.
    rx0, rw, ry0, ry1 = 9.3, 1.6, 0.5, 1.75
    robot_img = OUT_DIR / "robot.png"
    if robot_img.exists():
        box(ax, rx0, ry0, rw, ry1 - ry0, C_OUT, "")
        img = plt.imread(robot_img)
        ih, iw = img.shape[:2]
        h = ry1 - ry0 - 0.38
        w = min(h * iw / ih, rw - 0.1)
        h = w * ih / iw
        cx = rx0 + rw / 2
        ax.imshow(img, extent=(cx - w / 2, cx + w / 2, ry1 - 0.05 - h, ry1 - 0.05), zorder=3)
        ax.text(cx, ry0 + 0.15, "Franka + LEAP", ha="center", va="center", fontweight="bold",
                color=INK, zorder=4)
    else:
        box(ax, rx0, ry0 + 0.3, rw, ry1 - ry0 - 0.5, C_OUT, "Franka + LEAP",
            r"run first $T_a$, replan")
    arrow(ax, ((hx0 + hx1) / 2, hy0 - 0.27), ((hx0 + hx1) / 2, ry1 - (0 if robot_img.exists() else 0.2)),
          color=C_OUT)

    ax.text((fx + rx0 + rw) / 2, 0.3,
            "Train: DDPM noise-prediction loss, cosine schedule, AdamW     Inference: DDIM sampling",
            ha="center", va="center", fontsize=FS_SUB, color=MUTED)

    fig.savefig(OUT_DIR / "architecture.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_DIR / "architecture.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    print("wrote", OUT_DIR / "architecture.pdf", "and .png")


if __name__ == "__main__":
    main()
