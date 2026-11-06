"""Build a short Chinese learning note from completed, verified measurements."""
import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=ROOT / "run_1")
    args = parser.parse_args()
    result = json.loads((args.out_dir / "final_info.json").read_text(encoding="utf-8"))
    reload = json.loads((args.out_dir / "reload_check.json").read_text(encoding="utf-8"))
    stages = result["stages"]
    epochs = result["config"]["epochs"]
    rows = []
    for key, name in [("baseline", "作者预训练模型"), ("pruned", "刚剪枝的模型"), ("finetuned", "微调后的模型")]:
        value = stages[key]
        rows.append((name, f"{value['accuracy'] * 100:.2f}%", f"{value['params']:,}", f"{value['macs_per_image'] / 1e6:.3f}"))
    parameter_reduction = 100 * (1 - stages["pruned"]["params"] / stages["baseline"]["params"])
    mac_reduction = 100 * (1 - stages["pruned"]["macs_per_image"] / stages["baseline"]["macs_per_image"])
    recovery = 100 * (stages["finetuned"]["accuracy"] - stages["pruned"]["accuracy"])
    text = f"""# DepGraph 第一轮：本机已跑通的结果

本轮完成了真实 CIFAR-10 数据上的基线评估、结构化剪枝、{epochs}轮微调和独立进程重载。它证明当前环境能开展这条实验路线；完整论文复现仍需按原始配置继续核对版本、稀疏训练、训练轮数和多次重复。

| 模型阶段 | 测试准确率 | 参数个数 | 单张图片计算量（百万，MAC口径） |
|---|---:|---:|---:|
"""
    text += "\n".join("| " + " | ".join(row) + " |" for row in rows)
    text += f"""

参数减少 **{parameter_reduction:.2f}%**，计算量减少 **{mac_reduction:.2f}%**；计算量比值为 **{result['mac_reduction_ratio']:.3f}**。这不代表实际推理速度提高同样倍数，本轮没有测速度。

微调{epochs}轮后，准确率较刚剪枝时变化 **{recovery:+.2f}个百分点**。微调后的准确率是固定最后一轮结果，没有按测试集选最佳轮次。

## 这些步骤有什么用

1. **基线**：开始改模型前的参照物。这里加载作者已经训练好的权重，先确认模型和数据处理正确。
2. **剪枝**：真正删掉部分网络通道和相关连接，所以模型参数个数会下降。依赖图帮助相互连接的层一起调整，避免删完以后尺寸对不上。
3. **微调**：在模型删减结构后继续训练，让剩下的参数重新适应任务。精度能恢复多少必须靠实验观察。
4. **保存与重载**：确认结果可以再次打开使用。独立进程重载的固定8张真实图片输出完全一致：**{reload['identical_logits']}**。

## 本轮实验设置

- 数据：CIFAR-10，50,000张训练图片、10,000张测试图片，输入为3个颜色通道、32×32像素。
- 测试：三个阶段都使用全部10,000张测试图片。
- 训练：{epochs}轮遍历训练集，batch={result['config']['batch_size']}；沿用作者 drop_last=True，每轮实际使用 {result['history'][0]['train_seen']:,} 张，最后不足一个批次的图片略过。
- 剪枝：group_norm，不做稀疏训练；全局渐进剪枝，实际 {result['pruning_steps']} 步，目标计算量比2.11。
- 微调：SGD，学习率0.01、动量0.9、权重衰减0.0005，FP32；种子42。单次运行不能支持统计显著性结论。
- GPU：{result['gpu']}；本进程 PyTorch 峰值已分配显存 {result['peak_allocated_mib']:.1f} MiB；此值不包含其他程序、驱动和全部显卡占用。
- 软件：PyTorch {result['environment']['torch']}，torchvision {result['environment']['torchvision']}，torch-pruning发行元数据 {result['environment']['torch-pruning']}。
- 作者代码：v1.6.1，提交 {result['author_commit']}；这是后续工具版本，不能视为原始2023代码环境完全一致。

## 结果文件与图

- [参数和计算量对比](Figure_1.png)
- [准确率变化](Figure_2.png)
- [完整实验记录](final_info.json)
- [独立进程重载记录](reload_check.json)

## 下一步怎么学

先回答：为什么剪枝后参数少了？为什么精度可能下降？微调是否恢复了精度？为什么计算量减少不能直接说速度变快？

下一轮保持基线和评价方式一致，只改变一个因素，例如微调轮数。完整复现时再核对作者100轮微调与稀疏训练设置，不能把本轮结果与论文最终精度直接对齐。
"""
    (args.out_dir / "学习结果.md").write_text(text, encoding="utf-8")
    table = "".join("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    body = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DepGraph 第一轮学习结果</title>
<style>body{{margin:0;background:#f4f6f8;color:#1c2b39;font:17px/1.75 system-ui,'Microsoft YaHei',sans-serif}}main{{max-width:980px;margin:32px auto;padding:32px;background:white;border-radius:14px}}h1{{font-size:30px}}h2{{font-size:23px;margin-top:32px}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #dce3ea}}th{{background:#eaf1f8}}img{{width:100%;height:auto}}.lead{{padding:18px;background:#eaf3fa;border-left:5px solid #0072b2}}a{{color:#0068a5}}@media(max-width:640px){{main{{margin:0;padding:18px}}table{{font-size:12px}}td,th{{padding:6px}}}}</style><main>
<p>实验学习 · CIFAR-10 · 本机实测</p><h1>模型变小以后，还能认对图片吗？</h1>
<p class="lead">已完成：作者权重评估 → 依赖图剪枝 → {epochs}轮微调 → 独立进程重载。<br>本轮属于流程验证，完整论文结果仍需进一步复现。</p>
<table><thead><tr><th>模型阶段</th><th>准确率</th><th>参数个数</th><th>百万MAC口径计算量/张</th></tr></thead><tbody>{table}</tbody></table>
<h2>先看模型缩小了多少</h2><p>参数减少{parameter_reduction:.2f}%，计算量减少{mac_reduction:.2f}%。参数是模型中要保存和学习的数值；计算量是处理一张图片所需运算的估计。</p><img src="Figure_1.png" alt="原始、剪枝和微调模型的参数与计算量比较">
<h2>再看识别能力怎样变化</h2><p>每个阶段都在相同的10,000张测试图片上评价。微调是继续训练已剪枝的模型，让剩下的参数重新适应任务；本轮变化为{recovery:+.2f}个百分点。</p><img src="Figure_2.png" alt="相同测试集上三个阶段的准确率">
<h2>你应该学会的四件事</h2><ol><li>基线给后续改动提供参照。</li><li>结构化剪枝真正删除通道；相关层需要一起调整。</li><li>微调可能修复精度损失，但必须实测。</li><li>计算量比值{result['mac_reduction_ratio']:.3f}不等于推理加速倍数，本轮没有测速。</li></ol>
<h2>这次验证的边界</h2><p>只有一个随机种子、{epochs}轮微调、没有稀疏训练。输入为32×32彩色图片；训练集50,000张，每轮实际训练{result['history'][0]['train_seen']:,}张。它还不是工业缺陷检测，也没有做RK3568部署。独立进程重载固定8张真实图片的输出完全一致：{reload['identical_logits']}。</p>
<p><a href="学习结果.md">详细设置与学习说明</a> · <a href="final_info.json">原始结果</a> · <a href="reload_check.json">重载验证</a> · <a href="../README.md">运行方法</a></p></main></html>"""
    (args.out_dir / "学习结果.html").write_text(body, encoding="utf-8")
    print("Chinese learning note and offline HTML generated.")


if __name__ == "__main__":
    main()
