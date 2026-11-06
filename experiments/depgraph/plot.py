"""Plot measured results only; no latency or paper-accuracy claims."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=ROOT / "run_1")
    args = parser.parse_args()
    result = json.loads((args.out_dir / "final_info.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.size": 11, "font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False})
    stages = result["stages"]
    labels = ["原模型", "刚剪枝", f"微调{result['config']['epochs']}轮后"]
    colors = ["#0072B2", "#D55E00", "#009E73"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for axis, key, scale, label in zip(axes, ["params", "macs_per_image"], [1e6, 1e6], ["参数个数（百万）", "单张图片计算量（百万，MAC口径）"]):
        values = [stages[k][key] / scale for k in ["baseline", "pruned", "finetuned"]]
        bars = axis.bar(labels, values, color=colors)
        axis.set_ylabel(label)
        axis.set_ylim(0, max(values) * 1.2)
        axis.bar_label(bars, fmt="%.3f", padding=4)
    for extension in ["png", "pdf"]:
        fig.savefig(args.out_dir / f"Figure_1.{extension}", dpi=300)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6.5, 4), layout="constrained")
    values = [100 * stages[k]["accuracy"] for k in ["baseline", "pruned", "finetuned"]]
    axis.plot(range(3), values, marker="o", color="#0072B2", linewidth=2)
    axis.set_xticks(range(3), labels)
    axis.set_ylabel("全部10,000张测试图片的准确率（%）")
    axis.set_xlim(-.25, 2.25)
    axis.set_ylim(0, 105)
    for x, value in enumerate(values):
        axis.annotate(f"{value:.2f}%", (x, value), xytext=(0, 10), textcoords="offset points", ha="center")
    axis.grid(axis="y", alpha=.2)
    for extension in ["png", "pdf"]:
        fig.savefig(args.out_dir / f"Figure_2.{extension}", dpi=300)
    plt.close(fig)
    print(f"Figures saved to {args.out_dir}")


if __name__ == "__main__":
    main()
