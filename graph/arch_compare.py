import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 颜色
BASELINE_FACE, BASELINE_EDGE, BASELINE_TXT = "#F4EFEC", "#C0504D", "#7A3A34"
GREEN, SOT_FACE, SOT_EDGE, SOT_TXT = "#1F7A58", "#EAF5EF", "#2E8B57", "#155A40"
BLUE = "#2B5F8A"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig = plt.figure(figsize=(13, 9.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.25)
ax_top, ax_bot = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
ax_top.set_xlim(0, 1); ax_top.set_ylim(0, 1); ax_top.axis("off")
ax_bot.set_xlim(0, 1010); ax_bot.set_yscale("log"); ax_bot.set_ylim(2.5e2, 2.5e6)

# 辅助函数
def box(ax, cx, cy, w, h, text, face, edge, lw, fontsize, bold=False, tcolor="#222"):
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h, boxstyle="round,pad=0.008,rounding_size=0.018",
                       linewidth=lw, edgecolor=edge, facecolor=face, zorder=2)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, zorder=3, color=tcolor,
            fontweight="bold" if bold else "normal")
    return p

def arrow(ax, x1, y1, x2, y2, color, lw=1.8, label=None, ldx=0, ldy=0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 linewidth=lw, color=color, zorder=1))
    if label:
        ax.text((x1+x2)/2+ldx, (y1+y2)/2+ldy, label, ha="center", va="center", fontsize=9,
                color=color, zorder=4, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))

fig.suptitle("LLM personalization: cost per turn — Cuo-CoT vs SoT (cot_opt)", fontsize=18, fontweight="bold", y=0.98, color="#1a1a1a")

# --- 左侧 Cuo-CoT ---
ax_top.text(0.26, 0.965, "Cuo-CoT — motivating baseline", fontsize=13.5, fontweight="bold", ha="center", color=BASELINE_TXT)
ax_top.text(0.26, 0.935, "state re-derived from the full history, every turn", fontsize=9.5, ha="center", color=BASELINE_TXT)
hist = FancyBboxPatch((0.06, 0.38), 0.40, 0.50, boxstyle="round,pad=0.008,rounding_size=0.018",
                      linewidth=1.6, edgecolor=BASELINE_EDGE, facecolor=BASELINE_FACE, zorder=2)
ax_top.add_patch(hist)
ax_top.text(0.26, 0.865, "Conversation history", ha="center", va="center", fontsize=10.5, color=BASELINE_TXT, zorder=3)
for yy, lbl in zip([0.755, 0.68, 0.605, 0.52, 0.432], ["turn 1", "turn 2", "turn 3", "…  ⋮  …", "turn T"]):
    ax_top.add_patch(FancyBboxPatch((0.08, yy), 0.36, 0.075, boxstyle="round,pad=0.004,rounding_size=0.010",
                                    linewidth=1.0, edgecolor="#C9B8B4", facecolor="#FFFFFF", zorder=3))
    ax_top.text(0.26, yy + 0.0375, lbl, ha="center", va="center", fontsize=9, color="#8a6b66", zorder=4)
arrow(ax_top, 0.26, 0.38, 0.26, 0.29, BLUE, lw=1.8)
ax_top.text(0.46, 0.335, "re-read\nEVERY turn", ha="center", va="center", fontsize=9, color=BLUE, zorder=4,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="#B9CBD9", lw=1))
box(ax_top, 0.26, 0.20, 0.42, 0.16, "Single LLM call\ninfer implicit state → pick answer", BASELINE_FACE, BASELINE_EDGE, 1.6, 10, tcolor=BASELINE_TXT)
arrow(ax_top, 0.26, 0.12, 0.26, 0.045, BASELINE_EDGE)
ax_top.text(0.26, 0.055, "answer (a/b/c/d)", ha="center", fontsize=9.5, color=BASELINE_TXT, zorder=4,
            bbox=dict(boxstyle="round,pad=0.12", fc=BASELINE_FACE, ec=BASELINE_EDGE, lw=1))
ax_top.text(0.26, 0.018, "state is transient — thrown away after the call", ha="center", fontsize=8.5, style="italic", color="#9b8b86")

# --- 右侧 SoT (严格拉开间距，绝不重叠) ---
ax_top.text(0.74, 0.965, "SoT (cot_opt) — this project", fontsize=13.5, fontweight="bold", ha="center", color=SOT_TXT)
ax_top.text(0.74, 0.935, "state maintained once, then reused every turn", fontsize=9.5, ha="center", color=SOT_TXT)

box(ax_top, 0.74, 0.86, 0.26, 0.07, "New user message (turn t)", SOT_FACE, SOT_EDGE, 1.4, 9.5)

# 【核心修复：将 intent 和 state 中心点拉开，缩窄宽度】
box(ax_top, 0.58, 0.62, 0.18, 0.15, "intent_induce\nprev state + msg\n→ state delta", SOT_FACE, SOT_EDGE, 1.6, 9)
box(ax_top, 0.86, 0.62, 0.20, 0.22, "USER IMPLICIT STATE\n12 fields\n≤ 2500 chars", "#D6EDE1", GREEN, 2.2, 10, bold=True, tcolor=SOT_TXT)
ax_top.text(0.86, 0.505, "bounded — does not grow\nwith conversation length", ha="center", fontsize=8.5, color=GREEN, style="italic", zorder=5)

arrow(ax_top, 0.74, 0.825, 0.74, 0.705, GREEN)
arrow(ax_top, 0.70, 0.62, 0.75, 0.62, GREEN, label="new delta", ldx=0.0, ldy=0.08)
arrow(ax_top, 0.75, 0.55, 0.70, 0.55, GREEN, label="prev state", ldx=0.0, ldy=-0.08)

arrow(ax_top, 0.86, 0.51, 0.86, 0.315, GREEN)
box(ax_top, 0.74, 0.20, 0.30, 0.15, "LLM call: answer\nfrom state only", SOT_FACE, SOT_EDGE, 1.6, 9.5, tcolor=SOT_TXT)
arrow(ax_top, 0.74, 0.125, 0.74, 0.045, GREEN)
ax_top.text(0.74, 0.055, "answer (a/b/c/d)", ha="center", fontsize=9.5, color=SOT_TXT, zorder=4,
            bbox=dict(boxstyle="round,pad=0.12", fc=SOT_FACE, ec=SOT_EDGE, lw=1))
ax_top.text(0.74, 0.018, "input stays the same size regardless of length", ha="center", fontsize=8.5, style="italic", color=SOT_TXT)

ax_top.plot([0.47, 0.47], [0.02, 0.90], color="#DDDDDD", lw=1.4, zorder=0)
ax_top.text(0.47, 0.945, "vs", ha="center", fontsize=15, fontweight="bold", color="#BBBBBB")

# --- 底部图表 ---
turns = [1, 25, 50, 75, 100, 125, 150, 200, 300, 400, 500, 750, 1000]
cot_cost = [800 * t for t in turns]
sot_cost = [625 for _ in turns]
ax_bot.plot(turns, cot_cost, color=BASELINE_EDGE, lw=2.6, marker="o", ms=4, label="Cuo-CoT: re-reads full history each turn")
ax_bot.plot(turns, sot_cost, color=GREEN, lw=2.6, marker="s", ms=4, label="SoT (cot_opt): bounded state")
ax_bot.fill_between(turns, cot_cost, sot_cost, color="#F2D7D2", alpha=0.7, zorder=0)
ax_bot.set_ylabel("tokens processed per turn  (log)", fontsize=11, color="#333")
ax_bot.set_xlabel("conversation length (turns)", fontsize=11, color="#333")
ax_bot.grid(True, which="both", linestyle="--", alpha=0.35)
ax_bot.spines["top"].set_visible(False); ax_bot.spines["right"].set_visible(False)
ax_bot.tick_params(axis="both", labelsize=9, colors="#444")
ax_bot.annotate("≈ 1M tokens every turn\n(unbounded, grows with history)", xy=(1000, cot_cost[-1]), xytext=(700, 1.2e6),
                arrowprops=dict(arrowstyle="->", color=BASELINE_EDGE, lw=1.4), fontsize=9.5, color=BASELINE_TXT, ha="center")
ax_bot.annotate("≈ 625 tokens every turn\n(bounded snapshot)", xy=(1000, 625), xytext=(700, 1.6e3),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4), fontsize=9.5, color=SOT_TXT, ha="center")
ax_bot.legend(loc="upper left", fontsize=9.5, framealpha=0.95, edgecolor="#DDDDDD")
ax_bot.set_title("Per-turn inference cost stays flat for SoT as the conversation grows", fontsize=11.5, color="#333", loc="left", pad=10)

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("architecture_comparison_v2.png", dpi=170, bbox_inches="tight", facecolor="white")
print("成功保存 architecture_comparison_v2.png")